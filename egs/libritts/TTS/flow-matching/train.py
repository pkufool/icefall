#!/usr/bin/env python3
# Copyright         2024  Xiaomi Corp.        (authors: Wei Kang,
#                                                       Han Zhu)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import logging
import os
from functools import partial
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import optim
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from checkpoint import load_checkpoint, resume_checkpoint, save_checkpoint
from lhotse.cut import Cut
from lhotse.utils import fix_random_seed
from model import get_model
from optim import Eden, ScaledAdam
from tokenizer import Tokenizer
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tts_datamodule import TtsDataModule
from utils import get_adjusted_batch_count, prepare_input, set_batch_count

from icefall import diagnostics
from icefall.checkpoint import (
    remove_checkpoints,
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.hooks import register_inf_check_hooks
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    get_parameter_groups_with_lrs,
    setup_logger,
    str2bool,
)

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler]


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--fm-decoder-downsampling-factor",
        type=str,
        default="1,2,4,2,1",
        help="""
        Downsampling factor for each stack of encoder layers in zipformer.
        (for flow-matching decoder)
        """,
    )

    parser.add_argument(
        "--fm-decoder-num-layers",
        type=str,
        default="2,2,4,4,4",
        help="""
        Number of zipformer encoder layers per stack, comma separated.
        (for flow-matching decoder)
        """,
    )

    parser.add_argument(
        "--fm-decoder-cnn-module-kernel",
        type=str,
        default="31,15,7,15,31",
        help="""
        Sizes of convolutional kernels in convolution modules in each
        encoder stack: a single int (shared by all stacks) or comma-separated
        list. (for flow-matching decoder)
        """,
    )

    parser.add_argument(
        "--fm-decoder-feedforward-dim",
        type=int,
        default=1536,
        help="""
        Feedforward dimension of the zipformer encoder layers per stack: a single
        int (shared by all stacks). (for flow-matching decoder)
        """,
    )

    parser.add_argument(
        "--fm-decoder-num-heads",
        type=int,
        default=4,
        help="""
        Number of attention heads in the zipformer encoder layers: a single
        int (shared by all stacks). (for flow-matching decoder)
        """,
    )

    parser.add_argument(
        "--fm-decoder-dim",
        type=int,
        default=512,
        help="""
        Embedding dimension in encoder stacks: a single int (shared by all stacks).
        (for flow-matching decoder)
        """,
    )

    parser.add_argument(
        "--text-encoder-downsampling-factor",
        type=str,
        default="1",
        help="""
        Downsampling factor for each stack of encoder layers. (for text encoder)
        """,
    )

    parser.add_argument(
        "--text-encoder-num-layers",
        type=str,
        default="4",
        help="""
        Number of zipformer encoder layers per stack, comma separated. (for text encoder)
        """,
    )

    parser.add_argument(
        "--text-encoder-feedforward-dim",
        type=int,
        default=512,
        help="""
        Feedforward dimension of the zipformer encoder layers, per stack.
        (for text encoder)
        """,
    )

    parser.add_argument(
        "--text-encoder-cnn-module-kernel",
        type=str,
        default="9",
        help="""
        Sizes of convolutional kernels in convolution modules in each encoder stack:
        a single int (shared by all stacks) or comma-separated list. (for text encoder)
        """,
    )

    parser.add_argument(
        "--text-encoder-num-heads",
        type=int,
        default=4,
        help="""
        Number of attention heads in the zipformer encoder layers: a single int
        (shared by all stacks). (for text encoder)
        """,
    )

    parser.add_argument(
        "--text-encoder-dim",
        type=int,
        default=192,
        help="""
        Embedding dimension in encoder stacks: a single int. (for text encoder)
        """,
    )

    parser.add_argument(
        "--query-head-dim",
        type=int,
        default=32,
        help="""
        Query/key dimension per head in encoder stacks.
        (for both flow-matching decoder and text encoder)
        """,
    )

    parser.add_argument(
        "--value-head-dim",
        type=int,
        default=12,
        help="""
        Value dimension per head in encoder stacks.
        (for both flow-matching decoder and text encoder)
        """,
    )

    parser.add_argument(
        "--pos-head-dim",
        type=int,
        default=4,
        help="""
        Positional-encoding dimension per head in encoder stacks.
        (for both flow-matching decoder and text encoder)
        """,
    )

    parser.add_argument(
        "--pos-dim",
        type=int,
        default=48,
        help="""
        Positional-encoding embedding dimension.
        (for both flow-matching decoder and text encoder)
        """,
    )

    parser.add_argument(
        "--time-embed-dim",
        type=int,
        default=192,
        help="Embedding dimension of timestamps embedding.",
    )

    parser.add_argument(
        "--text-embed-dim",
        type=int,
        default=192,
        help="Embedding dimension of text embedding.",
    )

    parser.add_argument(
        "--token-type",
        type=str,
        default="bpe",
        choices=["phone", "bpe", "letter"],
        help="Input token type of TTS model",
    )

    parser.add_argument(
        "--token-file",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="""The file that contains information that maps tokens to ids,
        which is a text file with '{token} {token_id}' per line if type is
        char or phone, otherwise it is a bpe_model file.
        """,
    )


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=60,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="""Checkpoints of pre-trained models, will load it if not None
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="flow-matching/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--base-lr", type=float, default=0.02, help="The base learning rate."
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=7500,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=10,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        """,
    )

    parser.add_argument(
        "--lr-hours",
        type=float,
        default=0,
        help="""If positive, --epoch is ignored and it specifies the number of hours 
        that affects how rapidly the learning rate decreases.
        """,
    )

    parser.add_argument(
        "--ref-duration",
        type=float,
        default=50,
        help="Reference batch duration for purposes of adjusting batch counts for setting various "
        "schedules inside the model",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--print-diagnostics",
        type=str2bool,
        default=False,
        help="Accumulate stats on activations, print them and exit.",
    )

    parser.add_argument(
        "--inf-check",
        type=str2bool,
        default=False,
        help="Add hooks to check for infinite module outputs and gradients.",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=4000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 1.
        """,
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=30,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=200,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision training.",
    )

    parser.add_argument(
        "--feat-scale",
        type=float,
        default=0.1,
        help="The scale factor of fbank feature",
    )

    parser.add_argument(
        "--condition-drop-ratio",
        type=float,
        default=0.2,
        help="""
        The drop rate of text condition during training 
        (for classifier-free guidance).
        """,
    )

    add_model_arguments(parser)

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.

    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": None,
            "sample_rate": 24000,
            "frame_shift_ms": 256 / 24000 * 1000,
            "feat_dim": 100,  # num of mel bank
            "env_info": get_env_info(),
            "pad_id": 0,
        }
    )

    return params


def compute_fbank_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    features: Tensor,
    features_lens: Tensor,
    token_ids: List[List[int]],
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training.
      features:
        The target acoustic feature.
      features_lens:
        The number of frames of each utterance.
      token_ids:
        Input token ids that representing the transcripts.
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
    """

    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    batch_size, num_frames, _ = features.shape

    features = torch.nn.functional.pad(
        features, (0, 0, 0, num_frames - features.size(1))
    )  # (B, T, F)
    noise = torch.randn_like(features)  # (B, T, F)

    # Sampling t from uniform distribution
    if is_training:
        t = torch.rand(batch_size, 1, 1, device=device)
    else:
        t = (
            (torch.arange(batch_size, device=device) / batch_size)
            .unsqueeze(1)
            .unsqueeze(2)
        )
    with torch.set_grad_enabled(is_training):

        loss = model(
            tokens=token_ids,
            features=features,
            features_lens=features_lens,
            noise=noise,
            t=t,
            condition_drop_ratio=params.condition_drop_ratio,
        )

    assert loss.requires_grad == is_training
    info = MetricsTracker()
    num_frames = features_lens.sum().item()
    info["frames"] = num_frames
    info["loss"] = loss.detach().cpu().item() * num_frames

    return loss, info


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: Optimizer,
    scheduler: LRSchedulerType,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer.
      scheduler:
        The learning rate scheduler, we call step() every epoch.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      scaler:
        The scaler used for mix precision training.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
      rank:
        The rank of the node in DDP training. If no DDP is used, it should
        be set to 0.
    """
    model.train()
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    # used to track the stats over iterations in one epoch
    tot_loss = MetricsTracker()

    saved_bad_model = False

    def save_bad_model(suffix: str = ""):
        save_checkpoint(
            filename=params.exp_dir / f"bad-model{suffix}-{rank}.pt",
            model=model,
            model_avg=model_avg,
            params=params,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=0,
        )

    for batch_idx, batch in enumerate(train_dl):

        if batch_idx % 10 == 0:
            set_batch_count(model, get_adjusted_batch_count(params))

        if (
            params.valid_interval is None
            and batch_idx == 0
            and not params.print_diagnostics
        ) or (
            params.valid_interval is not None
            and params.batch_idx_train % params.valid_interval == 0
            and not params.print_diagnostics
        ):
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            logging.info(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
            )
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )

        params.batch_idx_train += 1

        batch_size = len(batch["text"])

        """
        inputs:
            {
                'features': (B x NumFrames x NumFeatures) float tensor
                'features_lens': (B, ) int tensor
                'audio': (B x NumSamples) float tensor
                'audio_lens': (B, ) int tensor
                'token_ids': List[List[int]]  # when return_token_ids=True
                'prompt': Dict of above all. # when return_prompt=True
            }
        """
        inputs = prepare_input(
            params=params,
            batch=batch,
            device=device,
            return_token_ids=True,
            return_feature=True,
        )

        try:
            with autocast(enabled=params.use_fp16):
                loss, loss_info = compute_fbank_loss(
                    params=params,
                    model=model,
                    features=inputs["features"],
                    features_lens=inputs["features_lens"],
                    token_ids=inputs["token_ids"],
                    is_training=True,
                )

            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            scaler.scale(loss).backward()

            scheduler.step_batch(params.batch_idx_train)
            # Use the number of hours of speech to adjust the learning rate
            if params.lr_hours > 0:
                scheduler.step_epoch(
                    params.batch_idx_train
                    * params.max_duration
                    * params.world_size
                    / 3600
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        except Exception as e:
            logging.info(f"Caught exception : {e}.")
            save_bad_model()
            raise

        if params.print_diagnostics and batch_idx == 5:
            return

        if (
            rank == 0
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.average_period == 0
        ):
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )

        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
        ):
            save_checkpoint_with_global_batch_idx(
                out_dir=params.exp_dir,
                global_batch_idx=params.batch_idx_train,
                model=model,
                model_avg=model_avg,
                params=params,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_dl.sampler,
                scaler=scaler,
                rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )
        if params.batch_idx_train % 100 == 0 and params.use_fp16:
            # If the grad scale was less than 1, try increasing it.    The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            cur_grad_scale = scaler._scale.item()

            if cur_grad_scale < 1024.0 or (
                cur_grad_scale < 4096.0 and params.batch_idx_train % 400 == 0
            ):
                scaler.update(cur_grad_scale * 2.0)
            if cur_grad_scale < 0.01:
                if not saved_bad_model:
                    save_bad_model(suffix="-first-warning")
                    saved_bad_model = True
                logging.warning(f"Grad scale is small: {cur_grad_scale}")
            if cur_grad_scale < 1.0e-05:
                save_bad_model()
                raise RuntimeError(
                    f"grad_scale is too small, exiting: {cur_grad_scale}"
                )

        if params.batch_idx_train % params.log_interval == 0:
            cur_lr = max(scheduler.get_last_lr())
            cur_grad_scale = scaler._scale.item() if params.use_fp16 else 1.0

            logging.info(
                f"Epoch {params.cur_epoch}, batch {batch_idx}, "
                f"global_batch_idx: {params.batch_idx_train}, batch size: {batch_size}, "
                f"loss[{loss_info}], tot_loss[{tot_loss}], "
                f"cur_lr: {cur_lr:.2e}, "
                + (f"grad_scale: {scaler._scale.item()}" if params.use_fp16 else "")
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )
                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.use_fp16:
                    tb_writer.add_scalar(
                        "train/grad_scale", cur_grad_scale, params.batch_idx_train
                    )

    loss_value = tot_loss["loss"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""

    model.eval()
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    # used to summary the stats over iterations
    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        """
        inputs:
            {
                'features': (B x NumFrames x NumFeatures) float tensor
                'features_lens': (B, ) int tensor
                'audio': (B x NumSamples) float tensor
                'audio_lens': (B, ) int tensor
                'token_ids': List[List[int]]  # when return_token_ids=True
                'prompt': Dict of above all. # when return_prompt=True
            }
        """
        inputs = prepare_input(
            params=params,
            batch=batch,
            device=device,
            return_token_ids=True,
            return_feature=True,
        )

        loss, loss_info = compute_fbank_loss(
            params=params,
            model=model,
            features=inputs["features"],
            features_lens=inputs["features_lens"],
            token_ids=inputs["token_ids"],
            is_training=False,
        )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


def tokenize_text(c: Cut, tokenizer):
    # SpeechSynthesisDataset needs normalized_text, see tts_datamodule for details
    if hasattr(c.supervisions[0], "normalized_text"):
        text = c.supervisions[0].normalized_text
    else:
        c.supervisions[0].normalized_text = c.supervisions[0].text
        text = c.supervisions[0].text
    token_ids, tokens = tokenizer.texts_to_token_ids([text], return_tokens=True)
    # return raw tokens as well for debug purpose
    c.supervisions[0].token_ids = token_ids[0]
    c.supervisions[0].tokens = tokens[0]

    # tokenize text in prompt
    if hasattr(c, "prompt"):
        if hasattr(c.prompt.supervisions[0], "normalized_text"):
            text = c.prompt.supervisions[0].normalized_text
        else:
            c.prompt.supervisions[0].normalized_text = c.prompt.supervisions[0].text
            text = c.prompt.supervisions[0].text
        token_ids, tokens = tokenizer.texts_to_token_ids([text], return_tokens=True)
        c.prompt.supervisions[0].token_ids = token_ids[0]
        c.prompt.supervisions[0].tokens = tokens[0]
    return c


def remove_short_and_long_utt(c: Cut):
    # Keep only utterances with duration between 1 second and 20 seconds
    # You should use ../local/display_manifest_statistics.py to get
    # an utterance duration distribution for your dataset to select
    # the threshold
    if c.duration < 1.0 or c.duration > 20.0:
        return False
    return True


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))

    assert (
        params.sampling_rate == params.sample_rate
    ), f"""sample_rate must be equal to sampling_rate of feature extractor,
    got {params.sample_rate} and {params.sampling_rate}."""

    assert (
        params.feat_dim == params.n_mels
    ), f"""feat_dim must be equal to
    n_mels of feature extractor, got {params.feat_dim} and {params.n_mels}."""

    assert (
        params.frame_shift_ms == params.frame_shift / params.sampling_rate * 1000
    ), f"""
    frame_shift_ms must be equal to frame_shift / sampling_rate * 1000,
    got {params.frame_shift_ms} and {params.frame_shift / params.sampling_rate * 1000}."""

    assert params.return_tokens, f"params.return_tokens should be True."

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    os.makedirs(f"{params.exp_dir}/fbank", exist_ok=True)

    logging.info("Training started")

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")
    tokenizer = Tokenizer(token_type=params.token_type, token_file=params.token_file)
    params.vocab_size = tokenizer.vocab_size

    params.device = device

    logging.info(params)

    logging.info("About to create model")

    model = get_model(params)
    if params.checkpoint is not None:
        logging.info(f"Loading pre-trained model from {params.checkpoint}")
        _ = load_checkpoint(filename=params.checkpoint, model=model, strict=True)
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of parameters : {num_param}")

    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    checkpoints = resume_checkpoint(params=params, model=model, model_avg=model_avg)

    model = model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(
            model,
            lr=params.base_lr,
            include_names=True,
        ),
        lr=params.base_lr,  # should have no effect
        clipping_scale=2.0,
    )

    assert params.lr_hours >= 0
    if params.lr_hours > 0:
        scheduler = Eden(optimizer, params.lr_batches, params.lr_hours)
    else:
        scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs)

    scaler = GradScaler(enabled=params.use_fp16, init_scale=1024.0)

    if checkpoints is not None:
        # load state_dict for optimizers
        if "optimizer" in checkpoints:
            logging.info("Loading optimizer state dict")
            optimizer.load_state_dict(checkpoints["optimizer"])

        # load state_dict for schedulers
        if "scheduler" in checkpoints:
            logging.info("Loading scheduler state dict")
            scheduler.load_state_dict(checkpoints["scheduler"])

        if "grad_scaler" in checkpoints:
            logging.info("Loading grad scaler state dict")
            scaler.load_state_dict(checkpoints["grad_scaler"])

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(
            512
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    libritts = TtsDataModule(args)

    if params.full_libri:
        train_cuts = libritts.train_all_shuf_cuts()
    else:
        train_cuts = libritts.train_clean_460_cuts()

    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        # We only load the sampler's state dict when it loads a checkpoint
        # saved in the middle of an epoch
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None

    _tokenize_text = partial(tokenize_text, tokenizer=tokenizer)

    train_cuts = train_cuts.filter(remove_short_and_long_utt)
    train_cuts = train_cuts.map(_tokenize_text)
    train_dl = libritts.train_dataloaders(
        train_cuts, sampler_state_dict=sampler_state_dict
    )

    dev_clean_cuts = libritts.dev_clean_cuts()
    dev_clean_cuts = dev_clean_cuts.map(_tokenize_text)
    valid_dl = libritts.dev_dataloaders(dev_clean_cuts)

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        logging.info(f"Start epoch {epoch}")

        if params.lr_hours == 0:
            scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        params.cur_epoch = epoch

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=valid_dl,
            scaler=scaler,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )

        if params.print_diagnostics:
            diagnostic.print_diagnostics()
            break

        filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
        save_checkpoint(
            filename=filename,
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )

        if rank == 0:
            if params.best_train_epoch == params.cur_epoch:
                best_train_filename = params.exp_dir / "best-train-loss.pt"
                copyfile(src=filename, dst=best_train_filename)

            if params.best_valid_epoch == params.cur_epoch:
                best_valid_filename = params.exp_dir / "best-valid-loss.pt"
                copyfile(src=filename, dst=best_valid_filename)

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def main():
    parser = get_parser()
    TtsDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    main()
