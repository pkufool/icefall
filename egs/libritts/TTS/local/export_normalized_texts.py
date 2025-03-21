#!/usr/bin/env python3
# Copyright    2024  Xiaomi Corp.        (authors: Wei Kang)
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
This script normalizes the text in supervision and exports to output file.

Usage example:

    python3 ./local/export_normalized_texts.py \
            --output data/texts.txt \
            --manifests data/fbank/libritts_cuts_with_tokens_train-all-shuf.jsonl.gz

"""

import argparse
import logging
from pathlib import Path

from lhotse import CutSet, load_manifest_lazy
from text_normalizer import custom_english_cleaners
from tqdm.auto import tqdm


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        help="Path to the output file",
    )

    parser.add_argument(
        "--manifests",
        type=Path,
        nargs="+",
        help="Path to the manifests",
    )

    return parser.parse_args()


def main():
    args = get_args()

    manifests = args.manifests
    assert len(manifests) > 0, "No manifests provided"

    assert manifests[0].is_file(), f"{manifests[0]} does not exist"
    cut_set = load_manifest_lazy(manifests[0])
    for i in range(1, len(manifests)):
        assert manifests[i].is_file(), f"{manifests[i]} does not exist"
        cut_set = cut_set + load_manifest_lazy(manifests[i])

    if not args.output.parent.is_dir():
        args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for cut in tqdm(cut_set):
            if hasattr(cut.supervisions[0], "normalized_text"):
                text = cut.supervisions[0].normalized_text
            else:
                text = cut.supervisions[0].text
            text = custom_english_cleaners(text)
            f.write(f"{text}\n")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
