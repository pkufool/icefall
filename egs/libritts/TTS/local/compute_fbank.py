#!/usr/bin/env python3
# Copyright    2021-2024  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Zengwei Yao,
#                                                       Wei Kang)
#              2024       The Chinese Univ. of HK  (authors: Zengrui Jin)
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
This file computes fbank features of the specific dataset.
The raw manifests (cuts without features) are written to data/manifests
The final cuts (with features) are written to data/fbank
"""

import argparse
import json
import logging
import os

from pathlib import Path

import torch
from tts_datamodule import TorchAudioFbank, TorchAudioFbankConfig
from lhotse import CutSet, LilcomChunkyWriter
from lhotse.audio import Recording
from lhotse.cut import MonoCut
from lhotse.supervision import SupervisionSegment
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import get_executor

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=24000,
        help="""Sampling rate of the audio for computing fbank, the default value for LibriTTS is 24000, audio files will be resampled if a different sample rate is provided""",
    )
    parser.add_argument(
        "--frame-shift",
        type=int,
        default=256,
        help="The frame shift in samples for the feature extraction.",
    )
    parser.add_argument(
        "--frame-length",
        type=int,
        default=1024,
        help="The frame length in samples for the feature extraction.",
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=100,
        help="The number of mel bins for the feature extraction.",
    )
    parser.add_argument(
        "--librispeech-manifest-dir",
        type=str,
        default="download/LibriSpeech-PC",
        help="""
        Path to the LibriSpeech-PC manifest files, it should contains test-clean.json,
        test-other.json, dev-clean.json, dev-other.json, train-clean-100.json,
        train-clean-360.json, train-other-500.json.
        """,
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="test-clean",
        help="""
        The subset name of LibriSpeech-PC, ${subset}.json should be in ${manifest-dir} directory.
        """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="libritts",
        help="""
        The dataset name, e.g. libritts, librispeech
        """,
    )

    return parser.parse_args()


def prepare_libritts(args):
    src_dir = Path("data/manifests")
    prefix = args.dataset
    suffix = args.suffix

    if (src_dir / f"{prefix}_cuts_{args.subset}.{suffix}").is_file():
        logging.info(f"{args.dataset} {args.subset} already exists - skipping.")
        return

    manifests = read_manifests_if_cached(
        dataset_parts=[args.subset],
        output_dir=src_dir,
        prefix=prefix,
        suffix=suffix,
    )
    assert manifests is not None

    assert len(manifests) == 1, (len(manifests), list(manifests.keys()), args.subset)

    for partition, m in manifests.items():
        cuts_filename = f"{prefix}_cuts_{partition}.{suffix}"
        logging.info(f"Processing {partition}")
        cut_set = CutSet.from_manifests(
            recordings=m["recordings"],
            supervisions=m["supervisions"],
        )
        cut_set.to_file(src_dir / cuts_filename)


def prepare_librispeech(args):
    output_dir = Path("data/manifests")
    prefix = args.dataset
    suffix = args.suffix
    librispeech_manifest_dir = Path(args.librispeech_manifest_dir)
    subset = args.subset
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    if (output_dir / f"{prefix}_cuts_{subset}.{suffix}").is_file():
        logging.info(f"{prefix}_cuts_{subset}.{suffix} already exists - skipping.")
        return

    with open(librispeech_manifest_dir / f"{subset}.json", "r") as f:
        cuts = []
        for line in f:
            item = json.loads(line)
            audio_path = Path(f"download/LibriSpeech/{item['audio_filepath']}")
            idx = audio_path.stem
            if not audio_path.is_file():
                logging.warning(f"No such file: {audio_path}")
                continue
            recording = Recording.from_file(audio_path)
            supervision = SupervisionSegment(
                id=idx,
                recording_id=idx,
                start=0.0,
                duration=item["duration"],
                channel=0,
                text=item["text"],
            )
            cut = MonoCut(
                id=idx,
                start=0.0,
                duration=item["duration"],
                channel=0,
                supervisions=[supervision],
                recording=recording,
            )
            cuts.append(cut)
        cut_set = CutSet.from_cuts(cuts)
        cut_set.to_file(output_dir / f"{prefix}_cuts_{subset}.{suffix}")


def compute_fbank(args):
    manifest_dir = Path("data/manifests")
    output_dir = Path("data/fbank")

    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    num_jobs = min(10, os.cpu_count())

    prefix = args.dataset
    suffix = args.suffix
    subset = args.subset

    config = TorchAudioFbankConfig(
        sampling_rate=args.sampling_rate,
        n_mels=args.n_mels,
        n_fft=args.frame_length,
        hop_length=args.frame_shift,
    )
    extractor = TorchAudioFbank(config)

    with get_executor() as ex:  # Initialize the executor only once.
        cuts_filename = f"{prefix}_cuts_{subset}.{suffix}"
        if (output_dir / cuts_filename).is_file():
            logging.info(f"{args.dataset} {subset} already exists - skipping.")
            return
        logging.info(f"Processing {subset}")

        cut_set = CutSet.from_file(manifest_dir / cuts_filename)
        if args.dataset == "librispeech" and args.sampling_rate != 16000:
            logging.info(f"Resampling audio to {args.sampling_rate}")
            cut_set = cut_set.resample(args.sampling_rate)

        cut_set = cut_set.compute_and_store_features(
            extractor=extractor,
            storage_path=f"{output_dir}/{prefix}_feats_{subset}",
            # when an executor is specified, make more partitions
            num_jobs=num_jobs if ex is None else 80,
            executor=ex,
            storage_type=LilcomChunkyWriter,
        )
        cut_set.to_file(output_dir / cuts_filename)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    args.suffix = "jsonl.gz"

    if args.dataset == "librispeech":
        prepare_librispeech(args)
    else:
        assert args.dataset == "libritts"
        prepare_libritts(args)

    compute_fbank(args)
