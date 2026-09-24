#!/usr/bin/env python3
# Copyright    2021-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Mingshuang Luo,
#                                                       Zengwei Yao,
#                                                       Daniel Povey)
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
"""
Usage:

export CUDA_VISIBLE_DEVICES="0,1,2,3"

# For non-streaming model training:
./zapformer/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zapformer/exp \
  --full-libri 1 \
  --max-duration 1000

# For streaming model training:
./zapformer/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zapformer/exp \
  --causal 1 \
  --full-libri 1 \
  --max-duration 1000

It supports training with:
  - transducer loss (default)
  - ctc loss
  - attention decoder loss
"""


import argparse
import copy
import logging
import warnings
import math
import os
import re
import uuid
from functools import partial
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union

import k2
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from atdataset import ATDataloader, Fbank
from attention_decoder import AttentionDecoderModel
from decoder import Decoder
from joiner import Joiner
from lhotse.utils import fix_random_seed
from model import AsrModel
# the try-pass blocks around imports are to reduce the chance of failures when running multiple code
# versions in parallel; later, these can be removed.
try:
    from batched_rubik import BatchedRubik
    # could also have done:
    # from rubik import Rubik
except:
    pass

from tqdm import tqdm

from variable_combined_scheduler import VariableCombinedLRScheduler
try:
    from variable_combined_scheduler import InterpCosineLRScheduler
    LRSchedulerType = VariableCombinedLRScheduler
except:
    pass

SchedulerType = "VariableCombinedLRScheduler"

from torch.optim.lr_scheduler import LambdaLR
from subsampling import Conv2dSubsampling
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from zapformer import Zapformer
from ssentencepiece import Ssentencepiece

from icefall import diagnostics
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
import torch.distributed as dist
from icefall.dist import cleanup_dist
from icefall.env import get_env_info
from icefall.err import raise_grad_scale_is_too_small_error
from alternating_spec_augment import AlternatingSpecAugment

from icefall.hooks import register_inf_check_hooks
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    get_parameter_groups_with_lrs,
    setup_logger,
    str2bool,
)
try:
    from icefall.utils import  dist_barrier
except:
    pass


def replace_punctuation_with_space(s: str) -> str:
    return re.sub(r"[,\.?!\"，。？！“”：:、<>《》\[\]{}【】;；]", "", s)


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


def lookup(params: AttributeDict, name: str):
    """
    Interprets numerical arguments in `params` by taking into account base-dim;
    also parses comma-separated lists of integers, turning them into tuples.
    If a particular attribute ending in "dim" is not present we look up
    the same name but ending in "factor", and multiply the elements by base_dim.
    """
    try:
        attr = getattr(params, name)
        try:
            attr = tuple(map(int, attr.split(",")))  # tuple of comma-separated ints
            if len(attr) == 1:
                attr = attr[0]
        except:
            pass  # leave attr as it is, e.g. a string.
        return attr
    except AttributeError as e:
        if name[-3:] != "dim":
            raise e
        try:
            attr = getattr(params,  name[:-3] + "multiple")
            if isinstance(attr, str):
                attr = tuple(map(int, attr.split(",")))  # tuple of ints
                base_dim = params.base_dim
                attr = tuple([i * base_dim for i in attr])
                if len(attr) == 1:
                    attr = attr[0]
            else:  # assume int.
                assert isinstance(attr, (int, float)), (name, attr)
                attr = attr * params.base_dim
            return attr
        except AttributeError as e:
            raise RuntimeError(f"cannot find or infer attribute {name} in params: {e}")




def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--num-encoder-layers",
        type=str,
        default="6,8,14,8",
        help="Number of zapformer encoder layers per stack, comma separated.",
    )

    parser.add_argument(
        "--downsampling-factor",
        type=str,
        default="1,2,4,2",
        help="Downsampling factor for each stack of encoder layers.",
    )

    parser.add_argument(
        "--base-dim",
        type=int,
        default=64,
        help="Dimension that, via multiples, defines the dimensions of the model."
    )

    parser.add_argument(
        "--embed-multiple",
        type=int,
        default=6,
        help="Output dimension of frontend, as multiple of base-dim; determines bypass dimensions in zapformer stacks and zapformer output dim.",
    )

    parser.add_argument(
        "--feedforward-multiple",
        type=str,
        default="4,4,3,4",
        help="Factor by which the feedforward hidden dim is greater than the encoder-dim, per stack: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--num-heads",
        type=str,
        default="4",
        help="Number of attention heads in the zapformer encoder layers, per stack: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--encoder-multiple",
        type=str,
        default="5,8,12,8",
        help="Factor by which encoder-dim is larger then base-dim, per encoder stack.",
    )

    parser.add_argument(
        "--query-head-dim",
        type=str,
        default="64",
        help="Query/key dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--value-head-dim",
        type=str,
        default="96",
        help="Value dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-head-dim",
        type=str,
        default="4",
        help="Position encoding dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--conv-params",
        type=str,
        default="31,31,15,31",
        help="Parameters per channel of convolution kernels",
    )

    parser.add_argument(
        "--decoder-multiple",
        type=int,
        default=8,
        help="Factor by which embedding dimension in the decoder model is larger than base-dim.",
    )

    parser.add_argument(
        "--joiner-multiple",
        type=int,
        default=12,
        help="""Dimension used in the joiner model.
        Outputs from the encoder and decoder model are projected
        to this dimension before adding.
        """,
    )

    parser.add_argument(
        "--attention-decoder-multiple",
        type=int,
        default=8,
        help="""Factor by which attention decoder dim is larger than base-dim""",
    )

    parser.add_argument(
        "--attention-decoder-num-layers",
        type=int,
        default=6,
        help="""Number of transformer layers used in attention decoder""",
    )

    parser.add_argument(
        "--attention-decoder-attention-multiple",
        type=int,
        default=8,
        help="""Determines attention dimension used in attention decoder""",
    )

    parser.add_argument(
        "--attention-decoder-num-heads",
        type=int,
        default=8,
        help="""Number of attention heads used in attention decoder""",
    )

    parser.add_argument(
        "--attention-decoder-feedforward-multiple",
        type=int,
        default=4,
        help="""Factor by which feedforward hidden dim in attention decoder is larger than attention-decoder-dim"""
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

    parser.add_argument(
        "--use-transducer",
        type=str2bool,
        default=True,
        help="If True, use Transducer head.",
    )

    parser.add_argument(
        "--use-ctc",
        type=str2bool,
        default=True,
        help="If True, use CTC head.",
    )

    parser.add_argument(
        "--use-attention-decoder",
        type=str2bool,
        default=False,
        help="If True, use attention-decoder head.",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate of input audio.",
    )


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--master-addr",
        type=str,
        help="Master node address for DDP training (used in multi-machine setup).",
    )

    parser.add_argument(
        "--local-rank-start",
        type=int,
        default=0,
        help="""Start rank of processes on the current machine, used in multi-machine
        setup, e.g., 0 for first machine, 8 for second).""",
    )

    parser.add_argument(
        "--local-world-size",
        type=int,
        help="""Number of processes (GPUs) on the current machine, used in
        multi-machine setup.""",
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
        "--num-real-epochs",
        type=int,
        default=30,
        help="Number of epochs to train, including number of copies; num-epochs will be <= this.",
    )

    parser.add_argument(
        "--max-copies",
        type=int,
        default=1,
        help="The num_copies to use in the dataloader on the last epoch (it rises linearly with step count from --min-copies)"
    )

    parser.add_argument(
        "--min-copies",
        type=int,
        default=1,
        help="The num_copies to use in the dataloader on the first epoch (it rises linearly with step count to --max-copies)"
    )

    parser.add_argument(
        "--batches-per-epoch",
        type=int,
        default=2200,
        help="Assumed number of batches per epoch for purposes of setting learning rate; only "
        "makes a difference during the first batch, after which an observed value is used.  This "
        "is the num batches where num_copies==1, i.e. on the first epoch"
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
        default="zapformer/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--training-sets",
        nargs="+",
        help="A list of ATDataloader training manifest files.",
    )

    parser.add_argument(
        "--training-weights",
        type=str,
        help="A comma-separated list of mux weights for training sets.",
    )

    parser.add_argument(
        "--epoch-hours",
        type=float,
        help="Number of hours to process in each epoch.",
    )

    parser.add_argument(
        "--validation-sets",
        nargs="*",
        help="A list of ATDataloader validation manifest files.",
    )

    parser.add_argument(
        "--validation-weights",
        type=str,
        help="A comma-separated list of mux weights for validation sets.",
    )

    parser.add_argument(
        "--max-duration",
        type=float,
        default=800.0,
        help="Maximum duration (in seconds) of each batch during training.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of samples in each batch during training.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of workers to use for each data loader.",
    )

    parser.add_argument(
        "--num-buckets",
        type=int,
        default=30,
        help="Number of buckets for ATDataloader streaming bucketing.",
    )

    parser.add_argument(
        "--use-noise-augment",
        type=str2bool,
        default=True,
        help="Whether to use noise augmentation for training.",
    )

    parser.add_argument(
        "--noise-list",
        type=str,
        default="data/tars/musan.lst",
        help="The noise manifest used for augmentation.",
    )

    parser.add_argument(
        "--use-speed-perturb",
        type=str2bool,
        default=True,
        help="Whether to use speed perturbation for training.",
    )

    parser.add_argument(
        "--use-volume-perturb",
        type=str2bool,
        default=True,
        help="Whether to use volume perturbation for training.",
    )

    parser.add_argument(
        "--base-lr", type=float, default=0.02, help="The base learning rate."
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=17500,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--ref-duration",
        type=float,
        default=600,
        help="Reference batch duration for purposes of adjusting batch counts for setting various "
        "schedules inside the model",
    )

    parser.add_argument(
        "--prune-range",
        type=int,
        default=5,
        help="The prune range for rnnt loss, it means how many symbols(context)"
        "we are using to compute the loss",
    )

    parser.add_argument(
        "--lm-scale",
        type=float,
        default=0.25,
        help="The scale to smooth the loss with lm "
        "(output of prediction network) part.",
    )

    parser.add_argument(
        "--am-scale",
        type=float,
        default=0.0,
        help="The scale to smooth the loss with am (output of encoder network) part.",
    )

    parser.add_argument(
        "--simple-loss-scale",
        type=float,
        default=0.5,
        help="To get pruning ranges, we will calculate a simple version"
        "loss(joiner is just addition), this simple loss also uses for"
        "training (as a regularization item). We will scale the simple loss"
        "with this parameter before adding to the final loss.",
    )

    parser.add_argument(
        "--aux-loss-scale",
        type=float,
        default=0.05,
        help="Scale on auxiliary losses that are defined in the model, such "
        "as cosine loss.",
    )

    parser.add_argument(
        "--ctc-loss-scale",
        type=float,
        default=0.2,
        help="Scale for CTC loss.",
    )

    parser.add_argument(
        "--attention-decoder-loss-scale",
        type=float,
        default=0.8,
        help="Scale for attention-decoder loss.",
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
        "--use-bf16",
        type=str2bool,
        default=False,
        help="Whether to use bf16 in AMP.",
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

        - warm_step: The warmup period that dictates the decay of the
              scale on pruned loss (for transducer).
              Expressed in terms of the "adjusted batch count", i.e. the
              normalized batch count after adjusting for changes in batch size.
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
            "valid_interval": 10000,
            # parameters for zapformer
            "feature_dim": 80,
            "subsampling_factor": 4,  # not passed in, this is fixed.
            # parameters for attention-decoder
            "ignore_id": -1,
            "label_smoothing": 0.1,
            "warm_step": 2000,
            "env_info": get_env_info(),
        }
    )

    return params


class ParamPlotter:
    def __init__(self,
                 model: Union[nn.Module, DDP],
                 tb_writer: Optional[SummaryWriter],
                 period: int = 50):
        if isinstance(model, DDP):
            model = model.module
        self.model = model
        self.tb_writer = tb_writer
        device = next(model.parameters()).device
        self.device = device
        self.period = period
        self.grad_proj = torch.tensor(0.0, device=device)

    def step(self, batch_idx_train: int):
        # in addition to plotting param_proj and grad_proj and grad_proj_sign every "period" steps,
        # plot grad_proj for the first 50 out of every 1000 steps; this will give us a sense of how
        # stable the oscillations are.
        dense_period = 1000
        dense_length = 50
        if batch_idx_train % self.period > 1 and batch_idx_train % dense_period > dense_length:
            return

        generator = torch.Generator(device=self.device)
        generator.manual_seed(1)


        with torch.no_grad():
            param_proj = torch.tensor(0.0, device=self.device)
            grad_proj = torch.tensor(0.0, device=self.device)
            for p in self.model.parameters():
                proj = torch.randn(p.shape, generator=generator, device=self.device)
                param_proj = param_proj + (p * proj).sum()
                try:
                    grad_proj = grad_proj + (p.grad * proj).sum()
                except AttributeError:
                    pass

        tb_writer = self.tb_writer
        def dump(proj: Tensor, name: str):
            proj_min = proj.clone()
            proj_max = proj.clone()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(proj_min, op=dist.ReduceOp.MIN)
                dist.all_reduce(proj_max, op=dist.ReduceOp.MAX)
                dist.all_reduce(proj, op=dist.ReduceOp.SUM)
                proj = proj / dist.get_world_size()
                proj_diff = proj_max - proj_min
                if tb_writer is not None:
                    tb_writer.add_scalar(name + '_diff', proj_diff.item(), batch_idx_train)
                if tb_writer is not None:
                    tb_writer.add_scalar(name, proj.item(), batch_idx_train)
        if batch_idx_train % self.period == 0:
            dump(param_proj, f'train/param_proj')
            self.grad_proj = grad_proj
        if batch_idx_train % self.period == 1 and tb_writer is not None:
           tb_writer.add_scalar('train/grad_same_sign', (grad_proj * self.grad_proj).sign(), batch_idx_train)
        if (batch_idx_train % dense_period < dense_length or batch_idx_train % self.period == 0) and tb_writer is not None:
            tb_writer.add_scalar('train/grad_proj', grad_proj, batch_idx_train)




def _to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))


def get_encoder_embed(params: AttributeDict) -> nn.Module:
    # encoder_embed converts the input of shape (N, T, num_features)
    # to the shape (N, (T - 7) // 2, encoder_dims).
    # That is, it does two things simultaneously:
    #   (1) subsampling: T -> (T - 7) // 2
    #   (2) embedding: num_features -> encoder_dims
    # In the normal configuration, we will downsample once more at the end
    # by a factor of 2, and most of the encoder stacks will run at a lower
    # sampling rate.
    encoder_embed = Conv2dSubsampling(
        in_channels=params.feature_dim,
        out_channels=lookup(params, "embed_dim"),
        causal=params.causal,
    )
    return encoder_embed


def get_encoder_model(params: AttributeDict) -> nn.Module:
    encoder = Zapformer(
        input_dim=lookup(params, "embed_dim"),
        output_downsampling_factor=2,
        downsampling_factor=lookup(params, "downsampling_factor"),
        num_encoder_layers=lookup(params, "num_encoder_layers"),
        encoder_dim=lookup(params, "encoder_dim"),
        query_head_dim=lookup(params, "query_head_dim"),
        value_head_dim=lookup(params, "value_head_dim"),
        pos_head_dim=lookup(params, "pos_head_dim"),
        num_heads=lookup(params, "num_heads"),
        feedforward_multiple=lookup(params, "feedforward_multiple"),
        conv_params=lookup(params, "conv_params"),
        causal=params.causal,
        chunk_size=lookup(params, "chunk_size"),
        left_context_frames=lookup(params, "left_context_frames"),
    )
    return encoder


def get_decoder_model(params: AttributeDict) -> nn.Module:
    decoder = Decoder(
        vocab_size=params.vocab_size,
        decoder_dim=lookup(params, "decoder_dim"),
        blank_id=params.blank_id,
        context_size=params.context_size,
    )
    return decoder


def get_joiner_model(params: AttributeDict) -> nn.Module:
    output_downsampling_factor = 2
    joiner = Joiner(
        encoder_dim=lookup(params, "embed_dim") * output_downsampling_factor,
        decoder_dim=lookup(params, "decoder_dim"),
        joiner_dim=lookup(params, "joiner_dim"),
        vocab_size=params.vocab_size,
    )
    return joiner


def get_attention_decoder_model(params: AttributeDict) -> nn.Module:
    output_downsampling_factor = 2
    decoder = AttentionDecoderModel(
        vocab_size=params.vocab_size,
        decoder_dim=lookup(params, "attention_decoder_dim"),
        num_decoder_layers=params.attention_decoder_num_layers,
        attention_dim=lookup(params, "attention_decoder_attention_dim"),
        num_heads=params.attention_decoder_num_heads,
        feedforward_dim=params.attention_decoder_feedforward_multiple * lookup(params, "attention_decoder_attention_dim"),
        memory_dim=lookup(params, "embed_dim") * output_downsampling_factor,
        sos_id=params.sos_id,
        eos_id=params.eos_id,
        ignore_id=params.ignore_id,
        label_smoothing=params.label_smoothing,
    )
    return decoder


def get_model(params: AttributeDict) -> nn.Module:
    assert params.use_transducer or params.use_ctc, (
        f"At least one of them should be True, "
        f"but got params.use_transducer={params.use_transducer}, "
        f"params.use_ctc={params.use_ctc}"
    )

    encoder_embed = get_encoder_embed(params)
    encoder = get_encoder_model(params)

    if params.use_transducer:
        decoder = get_decoder_model(params)
        joiner = get_joiner_model(params)
    else:
        decoder = None
        joiner = None

    if params.use_attention_decoder:
        attention_decoder = get_attention_decoder_model(params)
    else:
        attention_decoder = None

    output_downsampling_factor = 2
    model = AsrModel(
        encoder_embed=encoder_embed,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        attention_decoder=attention_decoder,
        encoder_dim=output_downsampling_factor * lookup(params, "embed_dim"),
        decoder_dim=lookup(params, "decoder_dim"),
        vocab_size=params.vocab_size,
        use_transducer=params.use_transducer,
        use_ctc=params.use_ctc,
        use_attention_decoder=params.use_attention_decoder,
    )
    return model


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: nn.Module = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[SchedulerType] = None,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_batch is positive, it will load the checkpoint from
    `params.exp_dir/checkpoint-{params.start_batch}.pt`. Otherwise, if
    params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.

    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The scheduler that we are using.
    Returns:
      Return a dict containing previously saved training info.
    """
    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    keys = [
        "best_train_epoch",
        "best_valid_epoch",
        "batch_idx_train",
        "best_train_loss",
        "best_valid_loss",
    ]
    for k in keys:
        params[k] = saved_params[k]

    if params.start_batch > 0:
        if "cur_epoch" in saved_params:
            params["start_epoch"] = saved_params["cur_epoch"]

    return saved_params


def _parse_float_list(s: Optional[str]) -> Optional[list]:
    if s is None:
        return None
    ans = [float(x) for x in s.split(",")]
    assert len(ans) > 0, s
    return ans


def filter_func(sample: Dict[str, Any], sp: Ssentencepiece, sample_rate: int) -> bool:
    T = sample["audio"].size(1) / sample_rate  # in seconds
    T = int(T * 100)  # in 10 ms units
    T = ((T - 7) // 2 + 1) // 2
    text = sample["text"]
    tokens = sp.encode(text)
    S = len(tokens)
    # filter empty text, too short or too long text, and too long audio
    if S == 0 or T - S <= 5 or T / S > 25:
        return False
    return True


def map_func(sample: Dict[str, Any]) -> Dict[str, Any]:
    text = replace_punctuation_with_space(sample["text"])
    # remove extra spaces and strip leading/trailing spaces
    text = re.sub(r"\s+", " ", text).strip()
    sample["text"] = text
    return sample


def save_checkpoint(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[SchedulerType] = None,
    sampler: Optional[Any] = None,
    scaler: Optional[GradScaler] = None,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer used in the training.
      sampler:
       The sampler for the training dataset.
      scaler:
        The scaler used for mix precision training.
    """
    if rank != 0:
        return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        model_avg=model_avg,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        scaler=scaler,
        rank=rank,
    )

    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = params.exp_dir / "best-train-loss.pt"
        copyfile(src=filename, dst=best_train_filename)

    if params.best_valid_epoch == params.cur_epoch:
        best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        copyfile(src=filename, dst=best_valid_filename)



def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: Ssentencepiece,
    batch: dict,
    is_training: bool,
    aux_loss_scale: float = 0.0,
    specaug: Optional[nn.Module] = None,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of Zapformer in our case.
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
    """
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    features = batch["feature"]
    # at entry, features is (N, T, C)
    assert features.ndim == 3
    features = features.to(device)

    feature_lens = batch["feature_lens"].to(device)

    batch_idx_train = params.batch_idx_train

    texts = batch["text"]
    y = sp.encode(texts, out_type=int)
    y = k2.RaggedTensor(y)

    if specaug is not None:
        with torch.amp.autocast('cuda', enabled=False):
            features = specaug(features.to(torch.float), feature_lens)


    if batch_idx_train % 50 == 0:
        logging.info(
            f"rng_state={torch.cuda.get_rng_state()}, features-sum={features.sum()}"
        )

    with torch.set_grad_enabled(is_training):
        simple_loss, pruned_loss, ctc_loss, attention_decoder_loss = model(
            x=features,
            x_lens=feature_lens,
            y=y,
            prune_range=params.prune_range,
            am_scale=params.am_scale,
            lm_scale=params.lm_scale,
            aux_loss_scale=aux_loss_scale,
        )

        loss = 0.0

        adjusted_batch_count = params.batch_idx_train
        warm_step = params.warm_step
        def warmup_schedule(scale, initial_factor):
            # geometric warmup schedules.
            warmup_factor = (1. if adjusted_batch_count >= warm_step else
                             initial_factor + (adjusted_batch_count / warm_step) * (1 - initial_factor))
            return scale * warmup_factor

        if params.use_transducer:
            simple_loss_scale = params.simple_loss_scale
            pruned_loss_scale = warmup_schedule(1.0, 0.05)
            loss += simple_loss_scale * simple_loss + pruned_loss_scale * pruned_loss

        if params.use_ctc:
            loss += params.ctc_loss_scale * ctc_loss

        if params.use_attention_decoder:
            loss += params.attention_decoder_loss_scale * attention_decoder_loss

    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nframes = (feature_lens // params.subsampling_factor).sum().item()
        info["frames"] = nframes

    # Note: We use reduction=sum while computing the loss.
    info["loss"] = loss.detach().cpu().item()
    if params.use_transducer:
        info["simple_loss"] = simple_loss.detach().cpu().item()
        info["pruned_loss"] = pruned_loss.detach().cpu().item()
    if params.use_ctc:
        info["ctc_loss"] = ctc_loss.detach().cpu().item()
    if params.use_attention_decoder:
        info["attn_decoder_loss"] = attention_decoder_loss.detach().cpu().item()

    return loss, info


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: Ssentencepiece,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        loss, loss_info = compute_loss(
            params=params,
            model=model,
            sp=sp,
            batch=batch,
            is_training=False,
        )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss

def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
    scheduler: SchedulerType,
    sp: Ssentencepiece,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    specaug: Optional[nn.Module] = None,
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
        The optimizer we are using.
      scheduler:
        The learning rate scheduler, we call step() every step.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      scaler:
        The scaler used for mix precision training.
      model_avg:
        The stored model averaged from the start of training.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
      rank:
        The rank of the node in DDP training. If no DDP is used, it should
        be set to 0.
    """
    model.train()

    tot_loss = MetricsTracker()

    saved_bad_model = False

    param_plotter = ParamPlotter(model, tb_writer, period=50)

    def get_scaler_scale():
        if params.use_autocast and scaler._scale is not None:
            return scaler._scale.item()
        else:
            return 1.0

    def save_bad_model(suffix: str = ""):
        save_checkpoint_impl(
            filename=params.exp_dir / f"bad-model{suffix}-{rank}.pt",
            model=model,
            model_avg=model_avg,
            params=params,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            rank=0,
        )

    for batch_idx, batch in enumerate(tqdm(train_dl, total=len(train_dl), dynamic_ncols=True)):
        if batch_idx % 10 == 0:
            set_batch_count(model, get_adjusted_batch_count(params))

        params.batch_idx_train += 1
        batch_size = len(batch["text"])

        try:
            with torch.amp.autocast('cuda',
                enabled=params.use_autocast, dtype=params.dtype
            ):
                loss, loss_info = compute_loss(
                    params=params,
                    model=model,
                    sp=sp,
                    batch=batch,
                    is_training=True,
                    specaug=specaug,
                    aux_loss_scale=get_scaler_scale() * params.aux_loss_scale * (0.25 if params.batch_idx_train > 2000 else 1.0),
                )
            # summary stats
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.
            scaler.scale(loss).backward()
            scheduler.set_batch(batch_idx)  # sets batch-count within the epoch, and sets the LRs.
            scaler.step(optimizer)
            scaler.update()
            param_plotter.step(params.batch_idx_train)
            optimizer.zero_grad()
        except Exception as e:
            logging.info(f"Caught exception: {e}.")
            save_bad_model()
            display_and_save_batch(batch, params=params, sp=sp)
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
                scaler=scaler,
                rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )

        if params.use_autocast:
            cur_grad_scale = get_scaler_scale()

            if cur_grad_scale < 0.01:
                if not saved_bad_model:
                    save_bad_model(suffix="-first-warning")
                    saved_bad_model = True
                    if not params.inf_check:
                        register_inf_check_hooks(model)
                logging.warning(f"Grad scale is small: {cur_grad_scale}")

            if cur_grad_scale < 1.0e-05:
                save_bad_model()
                raise_grad_scale_is_too_small_error(cur_grad_scale)

            # If the grad scale was less than 1, try increasing it.    The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            if (batch_idx % 25 == 0 and cur_grad_scale < 2.0 or
                batch_idx % 100 == 0 and cur_grad_scale < 8.0 or
                batch_idx % 400 == 0 and cur_grad_scale < 32.0):
                scaler.update(cur_grad_scale * 2.0)

        if batch_idx % params.log_interval == 0:
            cur_lr = max(scheduler.get_last_lr())
            cur_grad_scale = get_scaler_scale()

            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}, "
                + (f"grad_scale: {scaler._scale.item()}" if params.use_autocast else "")
            )


            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )

                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.use_autocast:
                    tb_writer.add_scalar(
                        "train/grad_scale", cur_grad_scale, params.batch_idx_train
                    )

        if (
            valid_dl is not None
            and batch_idx % params.valid_interval == 0
            and not params.print_diagnostics
        ):
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                sp=sp,
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


    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def setup_dist_for_multinode(
    global_rank: int,
    world_size: int,
    master_port: int,
    master_addr: Optional[str] = None,
) -> None:
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = (
            "localhost" if master_addr is None else str(master_addr)
        )
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(master_port)

    os.environ["RANK"] = str(global_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group("nccl", rank=global_rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(global_rank % torch.cuda.device_count())


def run(local_rank, world_size, args):
    """
    Args:
      local_rank:
        Rank of the process on the current node, passed by `mp.spawn()`.
      world_size:
        Total number of GPUs/processes over all nodes for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    global_rank = args.local_rank_start + local_rank
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(global_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    params = get_params()
    params.update(vars(args))

    # synchronize seeds.  important for parameter initialization to be consistent.
    fix_random_seed(params.seed)

    if world_size > 1:
        setup_dist_for_multinode(
            global_rank=global_rank,
            world_size=world_size,
            master_port=params.master_port,
            master_addr=params.master_addr,
        )
    # need torch.distributed.barrier() after fix_random_seed() as it fixes
    #  random seeds of all GPUs, not just the GPU of this process.
    dist_barrier()

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")

    if args.tensorboard and global_rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    logging.info(f"Device: {device}")

    sp = Ssentencepiece(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.sos_id = params.eos_id = sp.piece_to_id("<sos/eos>")
    params.vocab_size = sp.get_piece_size()

    if not params.use_transducer:
        if not params.use_attention_decoder:
            params.ctc_loss_scale = 1.0
        else:
            assert params.ctc_loss_scale + params.attention_decoder_loss_scale == 1.0, (
                params.ctc_loss_scale,
                params.attention_decoder_loss_scale,
            )

    if params.use_bf16:  # amp + bf16
        assert torch.cuda.is_bf16_supported(), "Your GPU does not support bf16!"
        assert not params.use_fp16, "You can only use either fp16 or bf16"
        params.dtype = torch.bfloat16
        params.use_autocast = True
    elif params.use_fp16:  # amp + fp16
        params.dtype = torch.float16
        params.use_autocast = True
    else:  # fp32
        params.dtype = torch.float32
        params.use_autocast = False

    logging.info(f"Using dtype={params.dtype}")
    logging.info(f"Use AMP={params.use_autocast}")

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if global_rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = BatchedRubik(
        get_parameter_groups_with_lrs(model, lr=params.base_lr, include_names=False),
        lr=params.base_lr,
        beta1=0.99,
    )

    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(
            512
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    training_sets = params.training_sets
    training_weights = _parse_float_list(params.training_weights)
    assert training_sets is not None and len(training_sets) > 0, (
        "--training-sets must be provided when using ATDataloader"
    )
    if training_weights is not None:
        assert len(training_weights) == len(training_sets), (
            len(training_weights),
            len(training_sets),
        )

    validation_sets = params.validation_sets
    validation_weights = _parse_float_list(params.validation_weights)
    if validation_weights is not None:
        assert validation_sets is not None
        assert len(validation_weights) == len(validation_sets), (
            len(validation_weights),
            len(validation_sets),
        )

    feature_extractor = Fbank(
        sample_rate=params.sample_rate,
        n_mels=params.feature_dim,
    )
    _filter_func = partial(filter_func, sp=sp, sample_rate=params.sample_rate)

    valid_dl = None
    if validation_sets is not None and len(validation_sets) > 0:
        valid_dl = ATDataloader(
            datasets=validation_sets,
            mux_weights=validation_weights,
            max_duration=params.max_duration,
            max_samples=params.max_samples,
            feature_extractor=feature_extractor,
            min_length=1.0,
            max_length=20.0,
            map_func=map_func,
            filter_func=_filter_func,
            sample_rate=params.sample_rate,
            use_noise_augment=False,
            noise_manifest=None,
            use_speed_perturb=False,
            use_volume_perturb=False,
            is_test=True,
            num_workers=params.num_workers,
            seed=params.seed + 777,
            num_buckets=params.num_buckets,
        )

    scaler = GradScaler(enabled=params.use_autocast, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])


    if True:
        # Work out copies_per_epoch
        copies_per_epoch = [ ]
        cur_real_epochs = 0
        progress_increment = 1.0 / (params.max_copies + 1 - params.min_copies)
        cur_progress = 0.0
        # go in backwards order to minimize rounding errors.
        for n in reversed(range(params.min_copies, params.max_copies + 1)):
            cur_progress += progress_increment
            target_real_epochs = int(0.5 + cur_progress * params.num_real_epochs) # + 0.5 to round up.
            while cur_real_epochs < target_real_epochs:
                copies_per_epoch.append(n)
                cur_real_epochs += n
        copies_per_epoch = list(reversed(copies_per_epoch))

    num_epochs = len(copies_per_epoch)
    logging.info(f"Num epochs = {len(copies_per_epoch)}, num-real-epochs={sum(copies_per_epoch)} vs target {params.num_real_epochs}")
    logging.info(f"Copies per epoch: {copies_per_epoch}")


    train_dl = ATDataloader(
            datasets=training_sets,
            epoch_hours=params.epoch_hours,
            max_duration=params.max_duration,
            sample_rate=params.sample_rate,
            max_samples=params.max_samples,
            num_copies=1,
            is_test=False,
    )

    params.batches_per_epoch = len(train_dl)
    logging.info(f"Num batches per epoch = {params.batches_per_epoch}")

    # this InterpCosineLRScheduler inherits from VariableCombinedLRScheduler.
    # this configuration is halfway between a linear function (1 to 0) and the conventional
    # cosine LR scheduler.  It decays to a minimum of 0.025.
    scheduler = InterpCosineLRScheduler(optimizer,
                                        min_factor=0.025,
                                        linear_scale=0.5,
                                        batches_per_epoch=[params.batches_per_epoch * n for n in copies_per_epoch])
    if (
        checkpoints
        and "scheduler" in checkpoints
        and checkpoints["scheduler"] is not None
    ):
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])


    for epoch in range(params.start_epoch, num_epochs + 1):
        # fix all random seeds before starting the dataloaders, as they require
        # all seeds to be synchronized, in particular for the sampler, which
        # uns in the main process and relies on the currently-set random seed
        # (in practice it's just the random module's
        # seed and possibly the numpy seed that really matter here.
        dist_barrier()
        fix_random_seed(params.seed + epoch - 1)
        dist_barrier()

        num_copies = copies_per_epoch[epoch - 1]
        logging.info(
            f"On epoch {epoch}, for dataloader: num_copies={num_copies}, "
            "this will affect num batches."
        )
        train_dl = ATDataloader(
            datasets=training_sets,
            epoch_hours=params.epoch_hours,
            mux_weights=training_weights,
            max_duration=params.max_duration,
            max_samples=params.max_samples,
            feature_extractor=feature_extractor,
            min_length=1.0,
            max_length=30.0,
            map_func=map_func,
            filter_func=_filter_func,
            sample_rate=params.sample_rate,
            use_noise_augment=params.use_noise_augment,
            noise_manifest=params.noise_list,
            use_speed_perturb=params.use_speed_perturb,
            use_volume_perturb=params.use_volume_perturb,
            num_copies=num_copies,
            is_test=False,
            num_workers=params.num_workers,
            seed=params.seed + 500 * epoch + global_rank,
            num_buckets=params.num_buckets,
        )

        if global_rank == 0:
            (model.module if isinstance(model, DDP) else model).encoder.compute_projection_overlap(verbose=True)  # for diagnostics

        seed = params.seed + 50 * epoch  + 512 * global_rank

        specaug = AlternatingSpecAugment(
            seed=seed,
        )  # otherwise use all default settings.

        if torch.cuda.is_available():
            with torch.cuda.device(local_rank):
                # set CUDA seed for "my GPU" in a rank-and-epoch-dependent way.
                # This is not not very important, it should just affect the
                # AddNoise() module in subsampling.py
                torch.cuda.manual_seed(seed)
        else:
            torch.manual_seed(seed)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)
            tb_writer.add_scalar("train/num_copies", num_copies, params.batch_idx_train)

        params.cur_epoch = epoch
        scheduler.set_epoch(epoch)

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sp=sp,
            train_dl=train_dl,
            valid_dl=valid_dl,
            scaler=scaler,
            tb_writer=tb_writer,
            specaug=specaug,
            world_size=world_size,
            rank=global_rank,
        )

        if params.print_diagnostics:
            d = diagnostic.print_diagnostics()
            filename = params.exp_dir / f"diagnostics-epoch-{params.cur_epoch}.pt"
            torch.save(d, filename)
            logging.info(f"Saved detailed diagnostics to {filename}")
            break

        save_checkpoint(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            rank=global_rank,
        )

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def display_and_save_batch(
    batch: dict,
    params: AttributeDict,
    sp: Ssentencepiece,
) -> None:
    """Display the batch statistics and save the batch into disk.

    Args:
      batch:
        A batch of data from ATDataloader.
      params:
        Parameters for training. See :func:`get_params`.
      sp:
        The BPE model.
    """
    filename = f"{params.exp_dir}/batch-{uuid.uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)

    features = batch["feature"]

    logging.info(f"features shape: {features.shape}")

    y = sp.encode(batch["text"], out_type=int)
    num_tokens = sum(len(i) for i in y)
    logging.info(f"num tokens: {num_tokens}")


def scan_pessimistic_batches_for_oom(
    model: Union[nn.Module, DDP],
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    sp: Ssentencepiece,
    params: AttributeDict,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 1 would cause OOM."
    )
    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]
        try:
            with torch.amp.autocast('cuda',
                enabled=params.use_autocast, dtype=params.dtype
            ):
                loss, _ = compute_loss(
                    params=params,
                    model=model,
                    sp=sp,
                    batch=batch,
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
            display_and_save_batch(batch, params=params, sp=sp)
            raise
        logging.info(
            f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
        )


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        if args.local_world_size is None:
            local_world_size = world_size
        else:
            local_world_size = args.local_world_size
        mp.spawn(run, args=(world_size, args), nprocs=local_world_size, join=True)
    else:
        run(local_rank=0, world_size=1, args=args)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
