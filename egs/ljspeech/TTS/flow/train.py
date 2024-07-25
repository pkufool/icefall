#!/usr/bin/env python3
# Copyright         2023  Xiaomi Corp.        (authors: Zengwei Yao)
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
import logging
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import copy
import math
import os

from lhotse import Fbank, FbankConfig
import librosa
import optim

import sentencepiece as spm
import k2
import numpy as np
import torch
from torch import Tensor
import torch.multiprocessing as mp
import torch.nn as nn
from lhotse.cut import Cut
from lhotse.utils import fix_random_seed
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tts_datamodule import LJSpeechTtsDataModule

from optim import Eden, ScaledAdam
from scaling import ScheduledFloat

from utils import plot_tensor, save_plot

from icefall import diagnostics
from icefall.checkpoint import load_checkpoint, save_checkpoint
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.hooks import register_inf_check_hooks
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    setup_logger,
    str2bool,
    get_parameter_groups_with_lrs,
)

from model import TtsModel

from zipformer import Zipformer2

from torchdyn.core import NeuralODE

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler]


def get_adjusted_batch_count(params: AttributeDict) -> float:
    # returns the number of batches we would have used so far if we had used the reference
    # duration.  This is for purposes of set_batch_count().
    return (
        params.batch_idx_train
        * (params.max_duration * params.world_size)
        / params.ref_duration
    )


def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    if isinstance(model, DDP):
        # get underlying nn.Module
        model = model.module
    for name, module in model.named_modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
        if hasattr(module, "name"):
            module.name = name


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--num-encoder-layers",
        type=str,
        default="2,2,3,4,3,2",
        help="Number of zipformer encoder layers per stack, comma separated.",
    )

    parser.add_argument(
        "--downsampling-factor",
        type=str,
        default="1,2,4,8,2,1",
        help="Downsampling factor for each stack of encoder layers.",
    )

    parser.add_argument(
        "--feedforward-dim",
        type=str,
        default="512,768,1024,1536,1024,768",
        help="Feedforward dimension of the zipformer encoder layers, per stack, comma separated.",
    )

    parser.add_argument(
        "--num-heads",
        type=str,
        default="4,4,4,8,4,4",
        help="Number of attention heads in the zipformer encoder layers: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--encoder-dim",
        type=str,
        default="192,256,384,512,384,256",
        help="Embedding dimension in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--query-head-dim",
        type=str,
        default="32",
        help="Query/key dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--value-head-dim",
        type=str,
        default="12",
        help="Value dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-head-dim",
        type=str,
        default="4",
        help="Positional-encoding dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-dim",
        type=int,
        default="48",
        help="Positional-encoding embedding dimension",
    )

    parser.add_argument(
        "--encoder-unmasked-dim",
        type=str,
        default="192,192,256,256,256,192",
        help="Unmasked dimensions in the encoders, relates to augmentation during training.  "
        "A single int or comma-separated list.  Must be <= each corresponding encoder_dim.",
    )

    parser.add_argument(
        "--cnn-module-kernel",
        type=str,
        default="31,31,15,15,15,31",
        help="Sizes of convolutional kernels in convolution modules in each encoder stack: "
        "a single int or comma-separated list.",
    )

    parser.add_argument(
        "--time-embed-dim",
        type=int,
        default=192,
        help="Embedding dimension of timestamps embedding.",
    )

    parser.add_argument(
        "--causal",
        type=str2bool,
        default=False,
        help="If True, use causal version of model.",
    )

    parser.add_argument(
        "--chunk-size",
        type=str,
        default="16,32,64,-1",
        help="Chunk sizes (at 50Hz frame rate) will be chosen randomly from this list during training. "
        " Must be just -1 if --causal=False",
    )

    parser.add_argument(
        "--left-context-frames",
        type=str,
        default="64,128,256,-1",
        help="Maximum left-contexts for causal training, measured in frames which will "
        "be converted to a number of chunks.  If splitting into chunks, "
        "chunk left-context frames will be chosen randomly from this list; else not relevant.",
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
        default=30,
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
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_256_punc/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--base-lr", type=float, default=0.045, help="The base learning rate."
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
        default=3.5,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        """,
    )

    parser.add_argument(
        "--ref-duration",
        type=float,
        default=600,
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

        - subsampling_factor:  The subsampling factor for the model.

        - encoder_dim: Hidden dim for multi-head attention model.

        - num_decoder_layers: Number of decoder layer of transformer decoder.

        - warm_step: The warmup period that dictates the decay of the
              scale on "simple" (un-pruned) loss.
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
            "valid_interval": 50,  # For the 100h subset, use 800
            # parameters for zipformer
            "frame_shift_ms": 256 / 22050 * 1000,
            "feat_dim": 80,  # num of mel bank
            "warm_step": 2000,
            "env_info": get_env_info(),
        }
    )

    return params


def _to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))


def load_checkpoint_if_available(
    params: AttributeDict, model: nn.Module
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.

    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
    Returns:
      Return a dict containing previously saved training info.
    """
    if params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    saved_params = load_checkpoint(filename, model=model)

    keys = [
        "best_train_epoch",
        "best_valid_epoch",
        "batch_idx_train",
        "best_train_loss",
        "best_valid_loss",
    ]
    for k in keys:
        params[k] = saved_params[k]

    return saved_params


def get_encoder_model(params: AttributeDict) -> nn.Module:
    encoder = Zipformer2(
        output_downsampling_factor=2,
        downsampling_factor=_to_int_tuple(params.downsampling_factor),
        num_encoder_layers=_to_int_tuple(params.num_encoder_layers),
        encoder_dim=_to_int_tuple(params.encoder_dim),
        encoder_unmasked_dim=_to_int_tuple(params.encoder_unmasked_dim),
        query_head_dim=_to_int_tuple(params.query_head_dim),
        pos_head_dim=_to_int_tuple(params.pos_head_dim),
        value_head_dim=_to_int_tuple(params.value_head_dim),
        pos_dim=params.pos_dim,
        num_heads=_to_int_tuple(params.num_heads),
        feedforward_dim=_to_int_tuple(params.feedforward_dim),
        cnn_module_kernel=_to_int_tuple(params.cnn_module_kernel),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=params.causal,
        chunk_size=_to_int_tuple(params.chunk_size),
        left_context_frames=_to_int_tuple(params.left_context_frames),
    )
    return encoder


def get_model(params: AttributeDict) -> nn.Module:
    encoder = get_encoder_model(params)
    model = TtsModel(
        encoder=encoder,
        in_embed_dim=_to_int_tuple(params.encoder_dim)[0],
        out_embed_dim=max(_to_int_tuple(params.encoder_dim)),
        feat_dim=params.feat_dim,
        in_feat_dim=params.feat_dim * 2,
        vocab_size=params.vocab_size,
    )
    return model


def pad_durations(durations: List[List[int]], num_frames: int) -> List[List[int]]:
    """
    Durations: a list containing, for each utterance, a list of BPE-symbol durations;
           it's OK if some are zero.
      num_frames: the number of frames in the batch of utterances, will be the largest
             num-frames of the batch.
    Returns:
         A modified list of durations which all sum up to num_frames.  We add
         a padding duration on the end of the batch to make up the number.
    """
    return [x + [num_frames - sum(x)] for x in durations]


def get_default_durations(y_lens: List[int], num_frames: int) -> List[List[int]]:
    """
    Returns default durations of symbols in utterances.
    """
    # Note: we may later want to pad the symbols with 0 or EOS or something,
    # as get_frame_pos would otherwise return a value one past the end.
    return pad_durations(
        [[num_frames // length] * length for length in y_lens], num_frames
    )


def get_transcript_pos(durations: List[List[int]], num_frames: int) -> Tensor:
    """
    Gets position in the transcript for each frame, i.e. the position
    in the symbol-sequence to look up.

    Returns:  a Tensor of shape (batch_size, num_frames)
    """
    batch_size = len(durations)
    ans = torch.zeros(batch_size, num_frames, dtype=torch.int64)
    for b in range(batch_size):
        this_dur = durations[b]
        cur_frame = 0
        for i, d in enumerate(this_dur):
            ans[b, cur_frame : cur_frame + d] = i
            cur_frame += d
        assert cur_frame == num_frames, (cur_frame, num_frames)
    return ans


def get_src_positions(
    durations: List[List[int]], default_durations: List[List[int]], num_frames: int
) -> Tensor:
    """
    Returns a Tensor of floats showing, for each frame in the batch, what the frame-index
    would be if the durations all had their default values.     Some of the frame-indexes
    will not be exact integers.

    Returns: a Tensor of shape (batch_size, num_frames) containing values 0 <= t <= num_frame.
    """
    batch_size = len(durations)
    ans = Tensor(batch_size, num_frames)
    for b in range(batch_size):
        this_dur = durations[b]
        this_default_dur = default_durations[b]
        assert len(this_dur) == len(this_default_dur)
        length = len(this_dur)
        cur_frame = 0  # frame with actual durations
        cur_src_frame = 0  # frame with default durations
        for d, dd in zip(this_dur, this_default_dur):
            for i in range(d):  # for each frame of the actual duration..
                ans[b, cur_frame] = cur_src_frame + i * (dd / d)
                cur_frame += 1
            cur_src_frame += dd
        assert cur_frame == num_frames and cur_src_frame == num_frames
    return ans


def get_initial_state(
    y: Tensor,
    transcript_pos: Tensor,
    embed: nn.Embedding,  # embedding of BPE pieces
    noise: Tensor,
    noise_scale: float = 1.0,
):  #  we can tune noise_scale for performance.  must match in train vs. test.
    """
      Returns the initial state of the flow, i.e. x(0), which is the phones all being equally-spaced
      and filling up the duration of the utterance.  The duration will presumably reflect the
      max duration of any utterance in the batch, which we can estimate using some crude method.

      Args:
         y: a Tensor of shape (batch_size, max_len) containing symbol sequences (BPE pieces), one for each
            utterance in the batch.  the symbol sequences may be BPE pieces.  Things like EOS don't
            matter much here, as long as it's consistent.
      transcript_pos: of shape (batch_size, num_frames), it is  as returned by get_transcript_pos
            given the default durations.  It contains the position in the transcript for each frame
             of the output.  Will be used to look up embeddings of bpe pieces.
        embed: an embedding module to turn the y values into embedding vectors.
        noise: a Tensor of Gaussian noise of shape (batch_size, num_frames, feat_dim)
    noise_scale: a scale on the noise, may be tuned to match the scale of the embeddings as these may
        be different in different models which might affect performance.

      Returns:
         x0:  a Tensor of shape (batch_size, num_frames, feat_dim)
    """
    (batch_size, num_frames, feat_dim) = noise.shape
    max_len = y.shape[-1]  # will probably include an EOS position.
    assert y.shape == (batch_size, max_len)
    transcript_embed = embed(y)
    assert transcript_embed.shape == (batch_size, max_len, feat_dim)
    assert transcript_pos.shape == (batch_size, num_frames)

    ans = torch.gather(
        transcript_embed,
        dim=1,
        index=transcript_pos.unsqueeze(-1).expand(batch_size, num_frames, feat_dim),
    )
    return ans + noise * noise_scale


def lookup(x: Tensor, positions: Tensor) -> Tensor:
    """
      Index x by frame indexes that may be non-integer, in which case we interpolate.
      In effect this does bilinear warping.

      Args:
            x: of shape (batch_size, num_frames, feat_dim), the input to be warped.
    positions: of shape (batch_size, num_frames), positions 0 <= t <= num_frames to
               look up or interpolate within x.
    """
    (batch_size, num_frames, feat_dim) = x.shape
    positions = positions.clamp(min=0, max=num_frames - 1.0e-05)

    positions_floor = torch.floor(positions)
    positions_rem = positions - positions_floor
    positions_floor = positions_floor.to(
        torch.int64
    )  # depending on torch version, could use int32.
    positions_ceil = positions_floor + 1

    positions_floor = positions_floor.clamp(max=num_frames - 1)
    positions_ceil = positions_ceil.clamp(max=num_frames - 1)

    # torch.set_printoptions(profile="full")

    def gather(x, p):
        return torch.gather(
            x, dim=1, index=p.unsqueeze(-1).expand(batch_size, num_frames, feat_dim)
        )

    return gather(x, positions_floor) * (1.0 - positions_rem).unsqueeze(-1) + gather(
        x, positions_ceil
    ) * positions_rem.unsqueeze(-1)


def interpolate(
    t: Tensor, x0: Tensor, x1: Tensor, src_positions: Tensor, dest_positions: Tensor
) -> Tensor:
    """
      Interpolate between x0 and x1 in a way that includes time-warping.
      Args:
          t: a Tensor of shape (batch_size, 1, 1); if t==0 this will return x0,
             if t==1 this will return x1.
         x0: (batch_size, num_frames, feat_dim), the value to return on t==0
         x1: (batch_size, num_frames, feat_dim), the value to return on t==1
    src_positions: of shape (batch_size, num_frames), for each frame-index in x1 this
             gives the corresponding frame in x0, these frame values are not necessarily
             integers but are in the range 0..num_frames.
    dest_positions: of shape (batch_size, num_frames), for each frame-index in x0 this
             gives the corresponding frame in x1, these frame values are not necessarily
             integers but are in the range 0..num_frames
    """
    assert not x0.dtype in [
        torch.float16,
        torch.bfloat16,
    ]  # taking a small delta would not work
    (batch_size, num_frames, feat_dim) = x1.shape

    default_positions = (
        torch.arange(num_frames, device=x1.device)
        .unsqueeze(0)
        .expand(batch_size, num_frames)
    )

    # x0_positions gives the frame indexes in x0.  at t == 0 this is just default_positions i.e. no warping.
    x0_positions = src_positions * t + default_positions * (1 - t)
    # x1_positions gives the frame indexes in x1.  at t == 1 this is just default_positions, i.e. no warping.
    x1_positions = dest_positions * (1 - t) + default_positions * t

    return (1 - t).unsqueeze(-1) * lookup(x0, x0_positions) + t.unsqueeze(-1) * lookup(
        x1, x1_positions
    )


def to_tensor(y: List[List[int]]):
    # pad with zeros
    length = max([len(l) for l in y])
    y = [l + [0] * (length - len(l)) for l in y]
    return torch.tensor(y, dtype=torch.int64)


def to_tensor_float(y: List[List[float]]):
    # pad with zeros
    length = max([len(l) for l in y])
    y = [l + [0] * (length - len(l)) for l in y]
    return torch.tensor(y, dtype=torch.float)


def get_conditions(
    embed: nn.Embedding,
    y: List[List[int]],
    durations: List[List[int]],
    x1: Tensor,
    print_tensor: bool = False,
) -> Tensor:
    """
     Returns (xt, ut), where xt is the neural network input value x(t) evaluted
        at the provided random times, and ut is the target for the loss,
        representing the flow direction.
     Args:
        embed: the embedding module, gives embeddings of dimension feat_dim, can be random.
            y: the transcripts (without EOS), indexed [batch][position]
    durations: the duration of each symbol in y.  Same shape as y.
            t: the (random) time values between 0 and 1.
        noise: gaussian random noise of shape (batch_size, num_frames, feat_dim)
           x1: the target speech, of shape (batch_size, num_frames, feat_dim)
    """
    (batch_size, num_frames, feat_dim) = x1.shape

    durs = pad_durations(durations, num_frames)

    transcript_pos = get_transcript_pos(durs, num_frames).to(x1.device)  # (B, T)

    y_padded = [l + [0] for l in y]
    y_padded = to_tensor(y_padded).to(x1.device)  # (B, S + 1)

    if print_tensor:
        torch.set_printoptions(profile="full")
        logging.info(f"durations : {durations[0]},  {len(durations[0])}")
        logging.info(f"y : {y[0]}, {len(y[0])}")
        logging.info(f"y_padded : {y_padded[0]}")
        logging.info(f"durs : {durs[0]}")
        logging.info(f"transcript_pos : {transcript_pos[0]}")

    max_len = y_padded.shape[-1]  # will probably include an EOS position.
    assert y_padded.shape == (batch_size, max_len)

    transcript_embed = embed(y_padded)  # (B, S + 1, F)

    assert transcript_embed.shape == (batch_size, max_len, feat_dim)
    assert transcript_pos.shape == (batch_size, num_frames)

    ans = torch.gather(
        transcript_embed,
        dim=1,
        index=transcript_pos.unsqueeze(-1).expand(batch_size, num_frames, feat_dim),
    )

    return ans


def get_duration_conditions(
    embed: nn.Embedding,
    y: List[List[int]],
    device,
) -> Tensor:
    """
     Returns (xt, ut), where xt is the neural network input value x(t) evaluted
        at the provided random times, and ut is the target for the loss,
        representing the flow direction.
     Args:
        embed: the embedding module, gives embeddings of dimension feat_dim, can be random.
            y: the transcripts (without EOS), indexed [batch][position]
    durations: the duration of each symbol in y.  Same shape as y.
            t: the (random) time values between 0 and 1.
        noise: gaussian random noise of shape (batch_size, num_frames, feat_dim)
           x1: the target speech, of shape (batch_size, num_frames, feat_dim)
    """

    batch_size = len(y)
    y_padded = to_tensor(y).to(device)
    max_len = y_padded.shape[-1]
    assert y_padded.shape == (batch_size, max_len)

    transcript_embed = embed(y_padded)  # (B, S, F)

    return transcript_embed


def get_xt_and_ut(
    embed: nn.Embedding,
    y: List[List[int]],
    durations: List[List[int]],
    t: Tensor,
    noise: Tensor,
    x1: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
     Returns (xt, ut), where xt is the neural network input value x(t) evaluted
        at the provided random times, and ut is the target for the loss,
        representing the flow direction.
     Args:
        embed: the embedding module, gives embeddings of dimension feat_dim, can be random.
            y: the transcripts (without EOS), indexed [batch][position]
    durations: the duration of each symbol in y.  Same shape as y.
            t: the (random) time values between 0 and 1.
        noise: gaussian random noise of shape (batch_size, num_frames, feat_dim)
           x1: the target speech, of shape (batch_size, num_frames, feat_dim)
    """
    (batch_size, num_frames, feat_dim) = x1.shape

    durs = pad_durations(durations, num_frames)
    default_durs = get_default_durations([len(x) for x in y], num_frames)

    assert len(durs) == len(default_durs), (len(durs), len(default_durs))

    y_padded = [l + [0] for l in y]

    x0 = get_initial_state(
        to_tensor(y_padded).to(x1.device),
        get_transcript_pos(default_durs, num_frames).to(x1.device),
        embed,
        noise,
    )

    src_positions = get_src_positions(durs, default_durs, num_frames).to(x1.device)
    dest_positions = get_src_positions(default_durs, durs, num_frames).to(x1.device)

    # taking a small delta would not work in half precision
    assert not x0.dtype in [torch.float16, torch.bfloat16]
    delta = 0.001
    xt = interpolate(t, x0, x1, src_positions, dest_positions)
    xt_delta = interpolate(t + delta, x0, x1, src_positions, dest_positions)
    ut = (
        xt_delta - xt
    ) / delta  # derivative w.r.t. t, obtained by finite-difference method.

    return xt, ut


def get_x0(
    params: AttributeDict,
    y: List[List[int]],
    embed: nn.Embedding,
    noise_scale: float = 1.0,
    duration_per_symbol: float = 0.2,
):
    num_frames = (
        max([len(x) for x in y]) * duration_per_symbol * 1000 / params.frame_shift_ms
    )
    num_frames = int(num_frames)

    default_durs = get_default_durations([len(x) for x in y], num_frames)

    y_padded = [l + [0] for l in y]

    noise = torch.randn((len(y), num_frames, params.feat_dim), device=params.device)

    x0 = get_initial_state(
        y=to_tensor(y_padded).to(params.device),
        transcript_pos=get_transcript_pos(default_durs, num_frames).to(params.device),
        embed=embed,
        noise=noise,
        noise_scale=noise_scale,
    )
    return x0


def prepare_input(
    params: AttributeDict,
    batch: dict,
    sp: spm.SentencePieceProcessor,
    device: torch.device,
):
    """Parse batch data"""
    features = batch["features"].to(device)
    features_lens = batch["features_lens"].to(device)

    texts = batch["text"]
    tokens = sp.encode(texts)

    durations = []
    log_durations = []
    for i, cut in enumerate(batch["cut"]):
        duration = []
        log_duration = []
        for x in cut.supervisions[0].durations:
            x = int(x * 1000 / params.frame_shift_ms)
            duration.append(x)
            log_duration.append(math.log(x))
        assert len(tokens[i]) == len(duration), (len(tokens[i]), len(duration))
        durations.append(duration)
        log_durations.append(log_duration)

    assert len(tokens) == len(durations), (len(tokens), len(durations))

    return features, features_lens, tokens, durations, log_durations


def get_time_shape(x):
    for n in range(1, x.ndim):
        x = x.narrow(n, 0, 1)
    return x


def compute_duration_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    log_durations: Tensor,
    tokens: List[List[int]],
    is_training: bool,
    rank: int = 0,
) -> Tuple[Tensor, MetricsTracker]:

    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    d1 = to_tensor_float(log_durations).unsqueeze(-1).to(device)  # (B, S, 1)

    batch_size, num_symbols, dur_dim = d1.shape

    d0 = torch.randn_like(d1)  # (B, S, 1)
    t = torch.rand_like(get_time_shape(d1))  # (B, 1, 1)

    dt = d1 * t + d0 * (1 - t)

    duration_conditions = get_duration_conditions(
        embed=model.module.embed if isinstance(model, DDP) else model.embed,
        y=tokens,
        device=device,
    )

    dt = torch.cat((dt, duration_conditions), dim=2)  # (B, S, F + 1)

    dut = d1 - d0  # (B, S, 1)

    with torch.set_grad_enabled(is_training):
        try:
            with autocast(enabled=params.use_fp16):
                if isinstance(model, DDP):
                    dvt = model.module.forward_duration(t, dt.permute(1, 0, 2)).permute(
                        1, 0, 2
                    )  # (B, S, 1)
                else:
                    dvt = model.forward_duration(t, dt.permute(1, 0, 2)).permute(
                        1, 0, 2
                    )  # (B, S, 1)
        except:  # noqa
            raise
        if rank == 0 and params.batch_idx_train % 100 == 0:
            logging.info(f"dt mean : {torch.mean(dt)}, stddev : {torch.std(dt)}")
            logging.info(f"dut mean : {torch.mean(dut)}, stddev : {torch.std(dut)}")
            logging.info(f"dvt mean : {torch.mean(dvt)}, stddev : {torch.std(dvt)}")
            logging.info(f"dut : {dut}, dvt : {dvt}")

        loss = torch.mean((dvt - dut) ** 2)

    assert loss.requires_grad == is_training
    info = MetricsTracker()
    info["duration_loss"] = loss.detach().cpu().item()
    info[
        "utterances"
    ] = 1  # actually number of batches, just to make tensorboard curve look smooth.
    return loss, info


def compute_fbank_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    features: Tensor,
    tokens: List[List[int]],
    durations: List[List[int]],
    is_training: bool,
    tb_writer: Optional[SummaryWriter] = None,
    rank: int = 0,
) -> Tuple[Tensor, MetricsTracker]:

    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    batch_size, num_frames, feat_dim = features.shape

    x1 = features  # (B, T, F)
    x0 = torch.randn_like(features)  # (B, T, F)
    t = torch.rand_like(get_time_shape(x1))  # (B, 1, 1)

    xt = x1 * t + x0 * (1 - t)

    conditions = get_conditions(
        embed=model.module.embed if isinstance(model, DDP) else model.embed,
        y=tokens,
        durations=durations,
        x1=x1,
        print_tensor=False,  # params.batch_idx_train % 100 == 0,
    )  # (B, T, F)

    xt = torch.cat((xt, conditions), dim=2)  # (B, T, 2*F)

    ut = x1 - x0  # (B, T, F)

    with torch.set_grad_enabled(is_training):
        try:
            with autocast(enabled=params.use_fp16):
                vt = model(t, xt.permute(1, 0, 2)).permute(1, 0, 2)  # (B, T, C)
        except:  # noqa
            raise
        if rank == 0 and params.batch_idx_train % 100 == 0 and tb_writer is not None:
            xt_ = torch.transpose(xt[0], 0, 1).detach().cpu()
            ut_ = torch.transpose(ut[0], 0, 1).detach().cpu()
            vt_ = torch.transpose(vt[0], 0, 1).detach().cpu()
            logging.info(f"xt mean : {torch.mean(xt)}, stddev : {torch.std(xt)}")
            logging.info(f"ut mean : {torch.mean(ut)}, stddev : {torch.std(ut)}")
            logging.info(f"vt mean : {torch.mean(vt)}, stddev : {torch.std(vt)}")
            tb_writer.add_image(
                f"xt_ut_vt/xt_",
                plot_tensor(xt_),
                global_step=params.batch_idx_train,
                dataformats="HWC",
            )
            save_plot(xt_, f"{params.exp_dir}/fbank/xt_{params.batch_idx_train}.png")

            tb_writer.add_image(
                f"xt_ut_vt/ut_",
                plot_tensor(ut_),
                global_step=params.batch_idx_train,
                dataformats="HWC",
            )
            save_plot(ut_, f"{params.exp_dir}/fbank/ut_{params.batch_idx_train}.png")

            tb_writer.add_image(
                f"xt_ut_vt/vt_",
                plot_tensor(vt_),
                global_step=params.batch_idx_train,
                dataformats="HWC",
            )
            save_plot(vt_, f"{params.exp_dir}/fbank/vt_{params.batch_idx_train}.png")

        loss = torch.mean((vt - ut) ** 2)

    assert loss.requires_grad == is_training
    info = MetricsTracker()
    info["loss"] = loss.detach().cpu().item()
    return loss, info


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    optimizer: Optimizer,
    scheduler: LRSchedulerType,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
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
      sp:
        Used to convert text to bpe tokens.
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
            params=params,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=0,
        )

    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1

        batch_size = len(batch["text"])

        features, features_lens, tokens, durations, log_durations = prepare_input(
            params=params, batch=batch, sp=sp, device=device
        )

        try:
            dur_loss, dur_loss_info = compute_duration_loss(
                params=params,
                model=model,
                log_durations=log_durations,
                tokens=tokens,
                is_training=True,
                rank=rank,
            )

            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + dur_loss_info

            scaler.scale(dur_loss).backward()

            loss, loss_info = compute_fbank_loss(
                params=params,
                model=model,
                features=features,
                durations=durations,
                tokens=tokens,
                is_training=True,
                rank=rank,
                tb_writer=tb_writer,
            )

            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            scaler.scale(loss).backward()

            scheduler.step_batch(params.batch_idx_train)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        except Exception as e:
            logging.info(f"Caught exception : {e}.")
            save_bad_model()
            raise

        if params.print_diagnostics and batch_idx == 5:
            return

        if params.batch_idx_train % 100 == 0 and params.use_fp16:
            # If the grad scale was less than 1, try increasing it.    The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            cur_grad_scale = scaler._scale.item()

            if cur_grad_scale < 8.0 or (
                cur_grad_scale < 32.0 and params.batch_idx_train % 400 == 0
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

        if (
            params.batch_idx_train % params.valid_interval == 0
            and not params.print_diagnostics
        ):
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                sp=sp,
                valid_dl=valid_dl,
                world_size=world_size,
                rank=rank,
                tb_writer=tb_writer,
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

    loss_value = tot_loss["loss"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def generate_samples(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    features: Tensor,
    tokens: List[List[int]],
    net_="normal",
    step: Optional[int] = 0,
    tb_writer: Optional[SummaryWriter] = None,
):
    """
    Args:
      model:
        represents the neural network that we want to generate samples from
      step: int
        represents the current step of training
    """
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    inner_model = model.module if isinstance(model, DDP) else model

    y_padded = to_tensor(tokens).to(device)

    with torch.no_grad():
        inner_model = copy.deepcopy(inner_model)
        t0 = torch.randn(y_padded.shape).unsqueeze(-1)  # (B, S, 1)
        num_symbols = t0.shape[1]

        duration_conditions = get_duration_conditions(
            embed=inner_model.embed, y=tokens, device=device
        )
        duration_conditions = duration_conditions.permute(1, 0, 2)

        duration_model_lambda = (
            lambda t, dt, *args, **kwargs: inner_model.forward_duration(
                t, torch.cat((dt, duration_conditions), dim=2)
            )
        )

        node = NeuralODE(duration_model_lambda, solver="euler", sensitivity="adjoint")
        traj = node.trajectory(
            t0.permute(1, 0, 2).to(device),
            t_span=torch.linspace(0, 1, 20).to(device),
        )
        gen_log_durations = traj[-1, :].view([num_symbols, -1]).permute(1, 0)

        gen_durations = torch.exp(gen_log_durations).ceil().int().tolist()

        logging.info(f"Generated durations : {gen_durations}")

        x0 = torch.randn_like(features)
        x1 = features

        conditions = get_conditions(
            embed=inner_model.embed, y=tokens, durations=gen_durations, x1=features
        )

        batch_size, num_frames, feat_dim = x0.shape
        conditions = conditions.permute(1, 0, 2)

        model_lambda = lambda t, xt, *args, **kwargs: inner_model(
            t, torch.cat((xt, conditions), dim=2)
        )

        node = NeuralODE(model_lambda, solver="euler", sensitivity="adjoint")
        traj = node.trajectory(
            x0.permute(1, 0, 2).to(device),
            t_span=torch.linspace(0, 1, 100).to(device),
        )
        gen_fbank = traj[-1, :].view([num_frames, -1, feat_dim]).permute(1, 0, 2)

        if tb_writer is not None:
            for i in range(batch_size):
                original = torch.transpose(x1[i], 0, 1).cpu()
                gen = torch.transpose(gen_fbank[i], 0, 1).cpu()
                start = torch.transpose(x0[i], 0, 1).cpu()
                tb_writer.add_image(
                    f"fbank_{i}/ground_truth_",
                    plot_tensor(original),
                    global_step=0,
                    dataformats="HWC",
                )
                save_plot(original, f"{params.exp_dir}/fbank/original_{i}.png")

                tb_writer.add_image(
                    f"fbank_{i}/generated_",
                    plot_tensor(gen),
                    global_step=params.batch_idx_train,
                    dataformats="HWC",
                )
                save_plot(
                    gen,
                    f"{params.exp_dir}/fbank/generated_{params.batch_idx_train}_{i}.png",
                )

                tb_writer.add_image(
                    f"fbank_{i}/x0_",
                    plot_tensor(start),
                    global_step=params.batch_idx_train,
                    dataformats="HWC",
                )
                save_plot(
                    start,
                    f"{params.exp_dir}/fbank/x0_{params.batch_idx_train}_{i}.png",
                )
        return gen_fbank


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
    rank: int = 0,
    tb_writer: Optional[SummaryWriter] = None,
) -> MetricsTracker:
    """Run the validation process."""

    model.eval()
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    # used to summary the stats over iterations
    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        features, features_lens, tokens, durations, log_durations = prepare_input(
            params=params, batch=batch, sp=sp, device=device
        )

        loss, loss_info = compute_duration_loss(
            params=params,
            model=model,
            log_durations=log_durations,
            tokens=tokens,
            is_training=False,
        )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info

        loss, loss_info = compute_fbank_loss(
            params=params,
            model=model,
            features=features,
            durations=durations,
            tokens=tokens,
            is_training=False,
        )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info

        if batch_idx == 0 and rank == 0 and tb_writer is not None:
            generate_samples(
                params=params,
                model=model,
                features=features,
                tokens=tokens,
                step=params.batch_idx_train,
                tb_writer=tb_writer,
            )

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


def scan_pessimistic_batches_for_oom(
    model: Union[nn.Module, DDP],
    train_dl: torch.utils.data.DataLoader,
    sp: spm.SentencePieceProcessor,
    optimizer: torch.optim.Optimizer,
    params: AttributeDict,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 1 would cause OOM."
    )
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]

        features, features_lens, tokens, durations, log_durations = prepare_input(
            params=params, batch=batch, sp=sp, device=device
        )

        try:
            dur_loss, dur_loss_info = compute_duration_loss(
                params=params,
                model=model,
                log_durations=log_durations,
                tokens=tokens,
                is_training=True,
            )
            dur_loss.backward()

            loss, loss_info = compute_fbank_loss(
                params=params,
                model=model,
                features=features,
                durations=durations,
                tokens=tokens,
                is_training=True,
            )
            loss.backward()

            optimizer.zero_grad()
        except Exception as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current "
                    "max_duration setting. We recommend decreasing "
                    "max_duration and trying again.\n"
                    f"Failing criterion: {criterion} "
                    f"(={crit_values[criterion]}) ..."
                )
            raise
        logging.info(
            f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
        )


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

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    os.makedirs(os.path.dirname(f"{params.exp_dir}/fbank"), exist_ok=True)

    logging.info("Training started")

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()
    params.device = device

    logging.info(params)

    logging.info("About to create model")

    model = get_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of parameters : {num_param}")

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(params=params, model=model)

    model = model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(model, lr=params.base_lr, include_names=True),
        lr=params.base_lr,  # should have no effect
        clipping_scale=2.0,
    )

    scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs)

    if checkpoints is not None:
        # load state_dict for optimizers
        if "optimizer" in checkpoints:
            logging.info("Loading optimizer state dict")
            optimizer.load_state_dict(checkpoints["optimizer"])

        # load state_dict for schedulers
        if "scheduler" in checkpoints:
            logging.info("Loading scheduler state dict")
            scheduler.load_state_dict(checkpoints["scheduler"])

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(
            512
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    ljspeech = LJSpeechTtsDataModule(args)

    train_cuts = ljspeech.train_cuts()

    def remove_short_and_long_utt(c: Cut):
        # Keep only utterances with duration between 1 second and 20 seconds
        # You should use ../local/display_manifest_statistics.py to get
        # an utterance duration distribution for your dataset to select
        # the threshold
        if c.duration < 1.0 or c.duration > 20.0:
            return False
        return True

    train_cuts = train_cuts.filter(remove_short_and_long_utt)
    train_dl = ljspeech.train_dataloaders(train_cuts)

    valid_cuts = ljspeech.valid_cuts()
    valid_dl = ljspeech.valid_dataloaders(valid_cuts)

    if False and not params.print_diagnostics:
        scan_pessimistic_batches_for_oom(
            model=model,
            train_dl=train_dl,
            sp=sp,
            optimizer=optimizer,
            params=params,
        )

    scaler = GradScaler(enabled=params.use_fp16, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        logging.info(f"Start epoch {epoch}")

        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        params.cur_epoch = epoch

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        train_one_epoch(
            params=params,
            model=model,
            sp=sp,
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
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )

        # if epoch % params.save_every_n == 0 or epoch == params.num_epochs:
        #    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
        #    save_checkpoint(
        #        filename=filename,
        #        params=params,
        #        model=model,
        #        optimizer=optimizer,
        #        scheduler=scheduler,
        #        sampler=train_dl.sampler,
        #        scaler=scaler,
        #        rank=rank,
        #    )
        #    if rank == 0:
        #        if params.best_train_epoch == params.cur_epoch:
        #            best_train_filename = params.exp_dir / "best-train-loss.pt"
        #            copyfile(src=filename, dst=best_train_filename)

        #        if params.best_valid_epoch == params.cur_epoch:
        #            best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        #            copyfile(src=filename, dst=best_valid_filename)

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def main():
    parser = get_parser()
    LJSpeechTtsDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


def test_wrapping():
    import matplotlib.pyplot as plt

    sp = spm.SentencePieceProcessor()
    sp.load("./data/lang_bpe_256_punc/bpe.model")
    vocab_size = sp.get_piece_size()
    feat_dim = 80
    embed = torch.nn.Embedding(vocab_size, 80).requires_grad_(False)

    cut = json.load(open("./data/spectrogram/test.json"))

    samples, sr = librosa.load(cut["recording"]["sources"][0]["source"])

    fbank = Fbank(FbankConfig(num_mel_bins=feat_dim, sampling_rate=sr))

    features = fbank.extract(samples=samples, sampling_rate=sr)
    num_frames = len(features)

    features = torch.tensor(features).unsqueeze(0).repeat(2, 1, 1)

    features[1, num_frames - 100 :, :] = math.log(1e-10)

    tokens = sp.encode(cut["supervisions"][0]["text"])
    durations = [int(x * 100) for x in cut["supervisions"][0]["custom"]["durations"]]
    noise = torch.randn(2, num_frames, feat_dim)

    fig = plt.figure(figsize=(10, 100))
    axes = fig.subplots(101, 2, sharex=True)
    for t in range(100):
        i = t
        t = torch.tensor([[t / 100], [t / 100]])
        xt, ut = get_xt_and_ut(
            embed=embed,
            y=[tokens, tokens[0:-20]],
            durations=[durations, durations[0:-20]],
            t=t,
            noise=noise,
            x1=features,
        )
        axes[i][0].imshow(xt[0].numpy().transpose())
        axes[i][1].imshow(ut[0].numpy().transpose())

    # axes[100][0].imshow(features[0].numpy().transpose())
    # axes[100][1].imshow(features[1].numpy().transpose())

    plt.savefig(f"fbank_wrapping_ut.png")


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    main()
    # test_wrapping()
