#!/usr/bin/env python3
# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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

(1) greedy search
./transducer_stateless/pretrained.py \
        --checkpoint ./transducer_stateless/exp/pretrained.pt \
        --lang-dir ./data/lang_char \
        --method greedy_search \
        /path/to/foo.wav \
        /path/to/bar.wav \

(1) beam search
./transducer_stateless/pretrained.py \
        --checkpoint ./transducer_stateless/exp/pretrained.pt \
        --lang-dir ./data/lang_char \
        --method beam_search \
        --beam-size 4 \
        /path/to/foo.wav \
        /path/to/bar.wav \

You can also use `./transducer_stateless/exp/epoch-xx.pt`.

Note: ./transducer_stateless/exp/pretrained.pt is generated by
./transducer_stateless/export.py
"""


import argparse
import logging
import math
from pathlib import Path
from typing import List

import kaldifeat
import torch
import torch.nn as nn
import torchaudio
from beam_search import beam_search, greedy_search
from conformer import Conformer
from decoder import Decoder
from joiner import Joiner
from model import Transducer
from torch.nn.utils.rnn import pad_sequence

from icefall.env import get_env_info
from icefall.lexicon import Lexicon
from icefall.utils import AttributeDict


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint. "
        "The checkpoint is assumed to be saved by "
        "icefall.checkpoint.save_checkpoint().",
    )

    parser.add_argument(
        "--lang-dir",
        type=str,
        help="""Path to lang.
        Used only when method is ctc-decoding.
        """,
    )

    parser.add_argument(
        "--method",
        type=str,
        default="greedy_search",
        help="""Possible values are:
          - greedy_search
          - beam_search
        """,
    )

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="+",
        help="The input sound file(s) to transcribe. "
        "Supported formats are those supported by torchaudio.load(). "
        "For example, wav and flac are supported. "
        "The sample rate has to be 16kHz.",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="Used only when --method is beam_search",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; "
        "2 means tri-gram",
    )
    parser.add_argument(
        "--max-sym-per-frame",
        type=int,
        default=3,
        help="""Maximum number of symbols per frame. Used only when
        --method is greedy_search.
        """,
    )

    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "sample_rate": 16000,
            # parameters for conformer
            "feature_dim": 80,
            "embedding_dim": 256,
            "subsampling_factor": 4,
            "attention_dim": 256,
            "nhead": 4,
            "dim_feedforward": 1024,
            "num_encoder_layers": 12,
            "vgg_frontend": False,
            "env_info": get_env_info(),
        }
    )
    return params


def get_encoder_model(params: AttributeDict) -> nn.Module:
    encoder = Conformer(
        num_features=params.feature_dim,
        output_dim=params.vocab_size,
        subsampling_factor=params.subsampling_factor,
        d_model=params.attention_dim,
        nhead=params.nhead,
        dim_feedforward=params.dim_feedforward,
        num_encoder_layers=params.num_encoder_layers,
        vgg_frontend=params.vgg_frontend,
    )
    return encoder


def get_decoder_model(params: AttributeDict) -> nn.Module:
    decoder = Decoder(
        vocab_size=params.vocab_size,
        embedding_dim=params.embedding_dim,
        blank_id=params.blank_id,
        context_size=params.context_size,
    )
    return decoder


def get_joiner_model(params: AttributeDict) -> nn.Module:
    joiner = Joiner(
        input_dim=params.vocab_size,
        inner_dim=params.embedding_dim,
        output_dim=params.vocab_size,
    )
    return joiner


def get_transducer_model(params: AttributeDict) -> nn.Module:
    encoder = get_encoder_model(params)
    decoder = get_decoder_model(params)
    joiner = get_joiner_model(params)

    model = Transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
    )
    return model


def read_sound_files(
    filenames: List[str], expected_sample_rate: float
) -> List[torch.Tensor]:
    """Read a list of sound files into a list 1-D float32 torch tensors.
    Args:
      filenames:
        A list of sound filenames.
      expected_sample_rate:
        The expected sample rate of the sound files.
    Returns:
      Return a list of 1-D float32 torch tensors.
    """
    ans = []
    for f in filenames:
        wave, sample_rate = torchaudio.load(f)
        assert sample_rate == expected_sample_rate, (
            f"expected sample rate: {expected_sample_rate}. "
            f"Given: {sample_rate}"
        )
        # We use only the first channel
        ans.append(wave[0])
    return ans


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.lang_dir = Path(args.lang_dir)

    params = get_params()
    params.update(vars(args))
    logging.info(f"{params}")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    lexicon = Lexicon(params.lang_dir)

    params.blank_id = 0
    params.vocab_size = max(lexicon.tokens) + 1

    logging.info("Creating model")
    model = get_transducer_model(params)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()
    model.device = device

    logging.info("Constructing Fbank computer")
    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = params.sample_rate
    opts.mel_opts.num_bins = params.feature_dim

    fbank = kaldifeat.Fbank(opts)

    logging.info(f"Reading sound files: {params.sound_files}")
    waves = read_sound_files(
        filenames=params.sound_files, expected_sample_rate=params.sample_rate
    )
    waves = [w.to(device) for w in waves]

    logging.info("Decoding started")
    features = fbank(waves)
    feature_lengths = [f.size(0) for f in features]

    features = pad_sequence(
        features, batch_first=True, padding_value=math.log(1e-10)
    )

    feature_lengths = torch.tensor(feature_lengths, device=device)

    with torch.no_grad():
        encoder_out, encoder_out_lens = model.encoder(
            x=features, x_lens=feature_lengths
        )

    num_waves = encoder_out.size(0)
    hyps = []
    msg = f"Using {params.method}"
    if params.method == "beam_search":
        msg += f" with beam size {params.beam_size}"
    logging.info(msg)
    for i in range(num_waves):
        # fmt: off
        encoder_out_i = encoder_out[i:i+1, :encoder_out_lens[i]]
        # fmt: on
        if params.method == "greedy_search":
            hyp = greedy_search(
                model=model,
                encoder_out=encoder_out_i,
                max_sym_per_frame=params.max_sym_per_frame,
            )
        elif params.method == "beam_search":
            hyp = beam_search(
                model=model, encoder_out=encoder_out_i, beam=params.beam_size
            )
        else:
            raise ValueError(f"Unsupported method: {params.method}")

        hyps.append([lexicon.token_table[i] for i in hyp])

    s = "\n"
    for filename, hyp in zip(params.sound_files, hyps):
        words = " ".join(hyp)
        s += f"{filename}:\n{words}\n\n"
    logging.info(s)

    logging.info("Decoding Done")


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
