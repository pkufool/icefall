#!/usr/bin/env python3
# Copyright         2024  Xiaomi Corp.        (authors: Wei Kang)
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
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scipy.io.wavfile import write
import json
import numpy as np

import sentencepiece as spm
import torch
import torch.nn as nn
from tts_datamodule import LJSpeechTtsDataModule

from train import (
    add_model_arguments,
    get_conditions,
    get_duration_conditions,
    get_model,
    get_params,
)

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import (
    AttributeDict,
    setup_logger,
    str2bool,
)

from torchdyn.core import NeuralODE

import sys

sys.path.append("./hifi-gan/")
from env import AttrDict
from models import Generator as HiFiGAN

LOG_EPS = math.log(1e-10)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--hifigan-config",
        type=str,
        help="Path to the configure file of hifigan",
    )

    parser.add_argument(
        "--hifigan-checkpoint",
        type=str,
        help="Path to the checkpoint of hifigan",
    )

    add_model_arguments(parser)

    return parser


def to_tensor(y: List[List[int]]):
    # pad with zeros
    length = max([len(l) for l in y])
    y = [l + [0] * (length - len(l)) for l in y]
    return torch.tensor(y, dtype=torch.int64)


def get_vocoder(params):
    with open(params.hifigan_config) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)

    vocoder.load_state_dict(
        torch.load(params.hifigan_checkpoint, map_location="cpu")["generator"]
    )
    return vocoder


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    vocoder: nn.Module,
    sp: spm.SentencePieceProcessor,
    batch: dict,
):
    """
    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict.
    """
    device = next(model.parameters()).device

    texts = batch["text"]
    cut_ids = [cut.id for cut in batch["cut"]]

    features = batch["features"]
    assert features.ndim == 3

    features = features.to(device)
    # at entry, feature is (N, T, C)

    tokens = sp.encode(texts)

    y_padded = to_tensor(tokens).to(device)

    t0 = torch.randn(y_padded.shape).unsqueeze(-1)  # (B, S, 1)
    num_symbols = t0.shape[1]

    duration_conditions = get_duration_conditions(
        embed=model.embed, y=tokens, device=device
    )
    duration_conditions = duration_conditions.permute(1, 0, 2)

    duration_model_lambda = lambda t, dt, *args, **kwargs: model.forward_duration(
        t, torch.cat((dt, duration_conditions), dim=2)
    )

    node = NeuralODE(duration_model_lambda, solver="euler", sensitivity="adjoint")
    traj = node.trajectory(
        t0.permute(1, 0, 2).to(device),
        t_span=torch.linspace(0, 1, 20).to(device),
    )
    gen_log_durations = traj[-1, :].view([num_symbols, -1]).permute(1, 0)

    gen_durations = torch.exp(gen_log_durations).ceil().int().tolist()

    x0 = torch.randn_like(features)
    x1 = features

    conditions = get_conditions(
        embed=model.embed, y=tokens, durations=gen_durations, x1=features
    )

    batch_size, num_frames, feat_dim = x0.shape
    conditions = conditions.permute(1, 0, 2)

    model_lambda = lambda t, xt, *args, **kwargs: model(
        t, torch.cat((xt, conditions), dim=2)
    )

    node = NeuralODE(model_lambda, solver="euler", sensitivity="adjoint")
    traj = node.trajectory(
        x0.permute(1, 0, 2).to(device),
        t_span=torch.linspace(0, 1, 100).to(device),
    )
    gen_fbank = traj[-1, :].view([num_frames, -1, feat_dim]).permute(1, 2, 0)

    audios = vocoder.forward(features.permute(0, 2, 1))

    # torch.set_printoptions(profile="full")

    for i in range(audios.shape[0]):
        audio = audios[i]
        v_min, v_max = audio.min(), audio.max()
        audio = (audio - v_min) / (v_max - v_min) * 2 - 1
        audio = audio.cpu().squeeze().numpy()
        write(f"{params.res_dir}/{cut_ids[i]}.wav", 22050, audio)


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    vocoder: nn.Module,
    sp: spm.SentencePieceProcessor,
    test_set: str,
):
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    with open(f"{params.res_dir}/{test_set}.scp", "w", encoding="utf8") as f:
        for batch_idx, batch in enumerate(dl):
            texts = batch["text"]
            cut_ids = [cut.id for cut in batch["cut"]]

            decode_one_batch(
                params=params,
                model=model,
                vocoder=vocoder,
                sp=sp,
                batch=batch,
            )

            assert len(texts) == len(cut_ids), (len(texts), len(cut_ids))

            for i in range(len(texts)):
                f.write(f"{cut_ids[i]}\t{texts[i]}\n")

            num_cuts += len(texts)

            if batch_idx % 50 == 0:
                batch_str = f"{batch_idx}/{num_batches}"

                logging.info(
                    f"batch {batch_str}, cuts processed until now is {num_cuts}"
                )


@torch.no_grad()
def main():
    parser = get_parser()
    LJSpeechTtsDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    params.res_dir = params.exp_dir / "generated_wavs"

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> and <unk> are defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
        elif params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg + 1
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )

    model = model.to(device)
    model.eval()

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    vocoder = get_vocoder(params)
    vocoder = vocoder.to(device)
    vocoder.eval()
    vocoder.remove_weight_norm()
    num_param = sum([p.numel() for p in vocoder.parameters()])
    logging.info(f"Number of vocoder parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    ljspeech = LJSpeechTtsDataModule(args)

    test_cuts = ljspeech.test_cuts()

    test_dl = ljspeech.test_dataloaders(test_cuts)

    test_sets = ["test"]
    test_dls = [test_dl]

    for test_set, test_dl in zip(test_sets, test_dls):
        decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            vocoder=vocoder,
            sp=sp,
            test_set=test_set,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
