#!/usr/bin/env python3
# Copyright         2023  Xiaomi Corp.        (authors: Zengwei Yao,
#                                                       Wen Kang)
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
This file generates the file that maps tokens to IDs.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict

try:
    from piper_phonemize import get_espeak_map
except Exception as ex:
    raise RuntimeError(
        f"{ex}\nPlease run\n"
        "pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html"
    )


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--token-type",
        type=str,
        default="phone",
        help="Path to the dict that maps the text tokens to IDs",
    )

    parser.add_argument(
        "--texts",
        type=Path,
        default=Path("data/texts.txt"),
        help="Path to the normalized texts of all training samples.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/lang_phone/tokens.txt"),
        help="Path to the dict that maps the text tokens to IDs",
    )

    return parser.parse_args()


def get_token2id_phone(filename: Path):
    """Get a dict that maps token to IDs, and save it to the given filename."""
    all_tokens = get_espeak_map()  # token: [token_id]
    all_tokens = {token: token_id[0] for token, token_id in all_tokens.items()}
    # sort by token_id
    all_tokens = sorted(all_tokens.items(), key=lambda x: x[1])

    with open(filename, "w", encoding="utf-8") as f:
        for token, token_id in all_tokens:
            f.write(f"{token} {token_id}\n")


def get_token2id_letter(filename: Path, texts: Path):
    all_tokens = set()
    with open(texts, "r", encoding="utf-8") as f:
        for line in f:
            all_tokens.update(list(line.strip()))
    if " " in all_tokens:
        all_tokens.remove(" ")
    if "_" in all_tokens:
        all_tokens.remove("_")
    if "^" in all_tokens:
        all_tokens.remove("^")
    if "$" in all_tokens:
        all_tokens.remove("$")
    all_tokens = sorted(all_tokens)
    with open(filename, "w", encoding="utf-8") as f:
        # Add special tokens
        # If you change these special tokens, please also change them in
        # tokenizer.py
        f.write(f"_ 0\n")  # padding
        f.write(f"^ 1\n")  # beginning of an utterance (bos)
        f.write(f"$ 2\n")  # end of an utterance (eos)
        f.write(f" 3\n")  # word separator (whitespace)
        for i, token in enumerate(all_tokens):
            f.write(f"{token} {i+4}\n")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    if args.token_type == "phone":
        get_token2id_phone(args.output)
    else:
        assert args.token_type == "letter", args.token_type
        get_token2id_letter(args.output, args.texts)
