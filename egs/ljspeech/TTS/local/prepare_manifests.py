#!/usr/bin/env python3
# Copyright    2024     Xiaomi Corp.        (authors: Wei Kang)
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


import logging
import os
from pathlib import Path

from lhotse import (
    CutSet,
    load_manifest,
)
from lhotse.audio import RecordingSet
from lhotse.supervision import SupervisionSet


def generate_ljspeech():
    src_dir = Path("data/manifests")
    output_dir = Path("data/manifests")

    prefix = "ljspeech"
    suffix = "jsonl.gz"
    partition = "all"

    recordings = load_manifest(
        src_dir / f"{prefix}_recordings_{partition}.{suffix}", RecordingSet
    )
    supervisions = load_manifest(
        src_dir / f"{prefix}_supervisions_{partition}.{suffix}", SupervisionSet
    )

    cuts_filename = f"{prefix}_cuts_{partition}_raw.{suffix}"
    if (output_dir / cuts_filename).is_file():
        logging.info(f"{cuts_filename} already exists - skipping.")
        return
    logging.info(f"Processing {partition}")
    cut_set = CutSet.from_manifests(
        recordings=recordings, supervisions=supervisions
    )

    cut_set.to_file(output_dir / cuts_filename)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    generate_ljspeech()
