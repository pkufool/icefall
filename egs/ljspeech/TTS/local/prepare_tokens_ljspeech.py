#!/usr/bin/env python3
# Copyright         2023-2024  Xiaomi Corp.        (authors: Zengwei Yao,
#                                                            Wei Kang)
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
This file reads the texts in given manifest and save the new cuts with specific tokens.
"""

import argparse
import logging
from pathlib import Path

from lhotse import CutSet, load_manifest


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--tokens-type",
        type=str,
        required=True,
        default="phone",
        help="The tokens type, valid values are phone and bpe",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_256_punc/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--manifests-in",
        type=str,
        default="data/spectrogram/ljspeech_cuts_all.jsonl.gz",
        help="Path to the input manifests.",
    )

    parser.add_argument(
        "--manifests-out",
        type=str,
        default="data/spectrogram/ljspeech_cuts_all_tokens.jsonl.gz",
        help="Path to the output manifests.",
    )

    return parser


def prepare_bpe_ljspeech(args):
    try:
        import sentencepiece as spm
    except ModuleNotFoundError as ex:
        raise RuntimeError(f"{ex}\nPlese run\n  pip install sentencepiece\n")

    cut_set = load_manifest(args.manifests_in)
    sp = spm.SentencePieceProcessor()
    sp.Load(args.bpe_model)

    new_cuts = []
    for cut in cut_set:
        # Each cut only contains one supervision
        assert len(cut.supervisions) == 1, (len(cut.supervisions), cut)
        text = cut.supervisions[0].normalized_text
        tokens = sp.encode(text, out_type=str)
        cut.tokens = tokens
        new_cuts.append(cut)

    new_cut_set = CutSet.from_cuts(new_cuts)
    new_cut_set.to_file(args.manifests_out)


def prepare_phone_ljspeech(args):
    try:
        import tacotron_cleaner.cleaners
    except ModuleNotFoundError as ex:
        raise RuntimeError(f"{ex}\nPlease run\n  pip install espnet_tts_frontend\n")

    try:
        from piper_phonemize import phonemize_espeak
    except ModuleNotFoundError as ex:
        raise RuntimeError(
            f"{ex}\nPlese run\n  pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html\n"
        )

    cut_set = load_manifest(args.manifests_in)

    new_cuts = []
    for cut in cut_set:
        # Each cut only contains one supervision
        assert len(cut.supervisions) == 1, (len(cut.supervisions), cut)
        text = cut.supervisions[0].normalized_text
        # Text normalization
        text = tacotron_cleaner.cleaners.custom_english_cleaners(text)
        # Convert to phonemes
        tokens_list = phonemize_espeak(text, "en-us")
        tokens = []
        for t in tokens_list:
            tokens.extend(t)
        cut.tokens = tokens
        new_cuts.append(cut)

    new_cut_set = CutSet.from_cuts(new_cuts)
    new_cut_set.to_file(args.manifests_out)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    parser = get_parser()
    args = parser.parse_args()

    if args.tokens_type == "phone":
        prepare_phone_ljspeech(args)
    elif args.tokens_type == "bpe":
        prepare_bpe_ljspeech(args)
    else:
        logging.error(
            f"Supported tokens type are phone and bpe, given {args.tokens_type}."
        )
        exit(-1)
