#!/usr/bin/env python3
# Copyright         2024  Xiaomi Corp.        (authors: Han Zhu,
#                                                       Wei Kang)
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


import copy
import logging
from pathlib import Path

from lhotse import CutSet, fix_random_seed, load_manifest_lazy
from tqdm.auto import tqdm


def prepare_prompts(librispeech_pc_file):
    output_dir = Path("data/fbank")
    prefix = "librispeech"
    suffix = "jsonl.gz"
    partition = "test-clean"

    target_speech_ids = []
    prompt_speech_ids = []
    with open(librispeech_pc_file, "r", encoding="utf-8") as file:
        for line in file:
            item = line.strip().split("\t")
            prompt_speech_ids.append(item[0])
            target_speech_ids.append(item[3])

    cut_map = dict()
    cut_set = load_manifest_lazy(output_dir / f"{prefix}_cuts_{partition}.{suffix}")
    for cut in cut_set:
        cut_id = cut.id
        cut_map[cut_id] = cut

    new_cuts = []
    for target_id, prompt_id in tqdm(zip(target_speech_ids, prompt_speech_ids)):
        target_cut = copy.deepcopy(cut_map[target_id])
        target_cut.prompt = copy.deepcopy(cut_map[prompt_id])
        target_cut.prompt.id = prompt_id + "-prompt-for-" + target_id
        new_cuts.append(target_cut)

    new_cut_set = CutSet.from_cuts(new_cuts)
    new_cut_set.to_file(output_dir / f"{prefix}_cuts_with_prompts_{partition}.{suffix}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    fix_random_seed(0)
    librispeech_pc_file = "testdata/librispeech_pc_test_clean_cross_sentence.lst"

    prepare_prompts(librispeech_pc_file)
