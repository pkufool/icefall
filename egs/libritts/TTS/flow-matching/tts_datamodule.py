# Copyright      2021  Piotr Żelasko
# Copyright      2022-2024  Xiaomi Corporation     (Authors: Mingshuang Luo,
#                                                            Zengwei Yao,
#                                                            Zengrui Jin,
#                                                            Wei Kang,
#                                                            Han Zhu)
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

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torchaudio

from lhotse import CutSet, load_manifest_lazy, validate
from lhotse.cut import Cut, MonoCut
from lhotse.dataset import (  # noqa F401 for PrecomputedFeatures
    DynamicBucketingSampler,
    PrecomputedFeatures,
    SimpleCutSampler,
)
from lhotse.dataset.collation import collate_audio
from lhotse.dataset.input_strategies import (
    BatchIO,
    OnTheFlyFeatures,
)
from lhotse.dataset.sampling.base import TimeConstraint
from lhotse.features.base import FeatureExtractor, register_extractor
from lhotse.utils import Seconds, compute_num_frames, fix_random_seed, ifnone

from torch.utils.data import DataLoader

from icefall.utils import str2bool


class MelSpectrogramFeatures(torch.nn.Module):
    def __init__(
        self,
        sampling_rate=24000,
        n_mels=100,
        n_fft=1024,
        hop_length=256,
        center=True,
        power=1,
    ):
        """
        Wrapper of torchaudio MelSpectrogram features.
        """
        super().__init__()

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=center,
            power=power,
        )

    def forward(self, audio):
        assert len(audio.shape) == 2
        mel = self.mel_spec(audio)
        logmel = mel.clamp(min=1e-7).log()
        return logmel


@dataclass
class TorchAudioFbankConfig:
    sampling_rate: int = 24000
    n_mels: int = 100
    n_fft: int = 1024
    hop_length: int = 256
    center: bool = True
    power: int = 1


@register_extractor
class TorchAudioFbank(FeatureExtractor):
    name = "TorchAudioFbank"
    config_type = TorchAudioFbankConfig

    def __init__(self, config):
        super().__init__(config=config)

    def _feature_fn(self, sample):
        fbank = MelSpectrogramFeatures(
            sampling_rate=self.config.sampling_rate,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            center=self.config.center,
            power=self.config.power,
        )
        return fbank(sample)

    @property
    def device(self) -> Union[str, torch.device]:
        return self.config.device

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.n_mels

    def extract(
        self,
        samples: np.ndarray,
        sampling_rate: int,
    ) -> torch.Tensor:
        # Check for sampling rate compatibility.
        expected_sr = self.config.sampling_rate
        assert sampling_rate == expected_sr, (
            f"Mismatched sampling rate: extractor expects {expected_sr}, "
            f"got {sampling_rate}"
        )
        samples = torch.from_numpy(samples)
        assert samples.ndim == 2, samples.shape
        assert samples.shape[0] == 1, samples.shape

        mel = self._feature_fn(samples).squeeze().t()

        assert mel.ndim == 2, mel.shape
        assert mel.shape[1] == self.config.n_mels, mel.shape

        num_frames = compute_num_frames(
            samples.shape[1] / sampling_rate, self.frame_shift, sampling_rate
        )

        if mel.shape[0] > num_frames:
            mel = mel[:num_frames]
        elif mel.shape[0] < num_frames:
            mel = mel.unsqueeze(0)
            mel = torch.nn.functional.pad(
                mel, (0, 0, 0, num_frames - mel.shape[1]), mode="replicate"
            ).squeeze(0)

        return mel.numpy()

    @property
    def frame_shift(self) -> Seconds:
        return self.config.hop_length / self.config.sampling_rate


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class SpeechSynthesisDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the speech synthesis task.
    Each item in this dataset is a dict of:

    .. code-block::

        {
            'audio': (B x NumSamples) float tensor
            'features': (B x NumFrames x NumFeatures) float tensor
            'audio_lens': (B, ) int tensor
            'features_lens': (B, ) int tensor
            'text': List[str] of len B  # when return_text=True
            'tokens': List[List[str]]  # when return_tokens=True
            'speakers': List[str] of len B  # when return_spk_ids=True
            'cut': List of Cuts  # when return_cuts=True
            'prompt': Dict of above all  # when return_prompt=True
        }
    """

    def __init__(
        self,
        cut_transforms: List[Callable[[CutSet], CutSet]] = None,
        feature_input_strategy: BatchIO = PrecomputedFeatures(),
        feature_transforms: Union[Sequence[Callable], Callable] = None,
        return_text: bool = True,
        return_tokens: bool = False,
        return_spk_ids: bool = False,
        return_cuts: bool = False,
        return_audio: bool = False,
        return_prompt: bool = False,
    ) -> None:
        super().__init__()

        self.cut_transforms = ifnone(cut_transforms, [])
        self.feature_input_strategy = feature_input_strategy

        self.return_text = return_text
        self.return_tokens = return_tokens
        self.return_spk_ids = return_spk_ids
        self.return_cuts = return_cuts
        self.return_audio = return_audio
        self.return_prompt = return_prompt

        if feature_transforms is None:
            feature_transforms = []
        elif not isinstance(feature_transforms, Sequence):
            feature_transforms = [feature_transforms]

        assert all(
            isinstance(transform, Callable) for transform in feature_transforms
        ), "Feature transforms must be Callable"
        self.feature_transforms = feature_transforms

    def __getitem__(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        validate_for_tts(cuts)

        for transform in self.cut_transforms:
            cuts = transform(cuts)

        features, features_lens = self.feature_input_strategy(cuts)

        for transform in self.feature_transforms:
            features = transform(features)

        batch = {
            "features": features,
            "features_lens": features_lens,
        }

        if self.return_audio:
            audio, audio_lens = collate_audio(cuts)
            batch["audio"] = audio
            batch["audio_lens"] = audio_lens

        if self.return_text:
            # use normalized text
            text = [cut.supervisions[0].normalized_text for cut in cuts]
            batch["text"] = text

        if self.return_tokens:
            tokens = [cut.tokens for cut in cuts]
            batch["tokens"] = tokens

        if self.return_spk_ids:
            batch["speakers"] = [cut.supervisions[0].speaker for cut in cuts]

        if self.return_cuts:
            batch["cut"] = [cut for cut in cuts]

        if self.return_prompt:
            prompt_cuts = CutSet.from_cuts(
                [MonoCut.from_dict(cut.prompt) for cut in cuts]
            )
            self.return_prompt = False
            batch["prompt"] = self.__getitem__(prompt_cuts)
            self.return_prompt = True

        return batch


@dataclass
class PromptTimeConstraint(TimeConstraint):
    """
    Represents a time-based constraint for sampler classes that is used for prompt-based TTS。
    """

    def measure_length(self, example: Cut) -> float:
        return example.duration + example.prompt["duration"]


class TtsDataModule:
    """
    DataModule for tts experiments.
    It assumes there is always one train and valid dataloader,
    but there can be multiple test dataloaders (e.g. LibriTTS test-clean
    and test-other).

    It contains all the common data pipeline modules used in TTS
    experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,
    - cut concatenation,
    - on-the-fly feature extraction

    This class should be derived for specific corpora used in TTS tasks.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="TTS data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )
        group.add_argument(
            "--full-libri",
            type=str2bool,
            default=False,
            help="""When enabled, use the entire LibriTTS training set.
            Otherwise, use the 460h clean subset.""",
        )
        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/fbank"),
            help="Path to directory with train/valid/test cuts.",
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=200.0,
            help="Maximum pooled recordings duration (seconds) in a "
            "single batch. You can reduce it if it causes CUDA OOM.",
        )
        group.add_argument(
            "--bucketing-sampler",
            type=str2bool,
            default=True,
            help="When enabled, the batches will come from buckets of "
            "similar duration (saves padding frames).",
        )
        group.add_argument(
            "--num-buckets",
            type=int,
            default=30,
            help="The number of buckets for the DynamicBucketingSampler"
            "(you might want to increase it for larger datasets).",
        )
        group.add_argument(
            "--frame-shift",
            type=int,
            default=256,
            help="The frame shift in samples for the feature extraction.",
        )
        group.add_argument(
            "--frame-length",
            type=int,
            default=1024,
            help="The frame length in samples for the feature extraction.",
        )
        group.add_argument(
            "--n-mels",
            type=int,
            default=80,
            help="The number of mel bins for the feature extraction.",
        )
        group.add_argument(
            "--sampling-rate",
            type=int,
            default=24000,
            help="The sampling rate of the audio files.",
        )
        group.add_argument(
            "--on-the-fly-feats",
            type=str2bool,
            default=False,
            help="When enabled, use on-the-fly cut mixing and feature "
            "extraction. Will drop existing precomputed feature manifests "
            "if available.",
        )
        group.add_argument(
            "--shuffle",
            type=str2bool,
            default=True,
            help="When enabled (=default), the examples will be "
            "shuffled for each epoch.",
        )
        group.add_argument(
            "--drop-last",
            type=str2bool,
            default=True,
            help="Whether to drop last batch. Used by sampler.",
        )
        group.add_argument(
            "--return-cuts",
            type=str2bool,
            default=True,
            help="When enabled, each batch will have the "
            "field: batch['cut'] with the cuts that "
            "were used to construct it.",
        )
        group.add_argument(
            "--return-text",
            type=str2bool,
            default=True,
            help="When enabled, each batch will have the "
            "field: batch['text'] with the normalized "
            "text of the supervisions.",
        )
        group.add_argument(
            "--return-tokens",
            type=str2bool,
            default=True,
            help="When enabled, each batch will have the "
            "field: batch['tokens'] with the tokenized "
            "text of the supervisions.",
        )
        group.add_argument(
            "--return-spk-ids",
            type=str2bool,
            default=False,
            help="When enabled, each batch will have the "
            "field: batch['speakers'] with the speaker "
            "IDs of the supervisions.",
        )
        group.add_argument(
            "--return-audio",
            type=str2bool,
            default=False,
            help="When enabled, each batch will have the "
            "field: batch['audio'] with the audio samples.",
        )
        group.add_argument(
            "--return-prompt",
            type=str2bool,
            default=False,
            help="When enabled, each batch will have the "
            "field: batch['prompt'] with the prompt "
            "features, text, tokens, speakers, and audio.",
        )
        group.add_argument(
            "--num-workers",
            type=int,
            default=8,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )
        group.add_argument(
            "--input-strategy",
            type=str,
            default="PrecomputedFeatures",
            help="AudioSamples or PrecomputedFeatures",
        )

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
    ) -> DataLoader:
        """
        Args:
          cuts_train:
            CutSet for training.
          sampler_state_dict:
            The state dict for the training sampler.
        """
        logging.info("About to create train dataset")
        train = SpeechSynthesisDataset(
            return_text=self.args.return_text,
            return_tokens=self.args.return_tokens,
            return_spk_ids=self.args.return_spk_ids,
            feature_input_strategy=eval(self.args.input_strategy)(),
            return_cuts=self.args.return_cuts,
            return_audio=self.args.return_audio,
        )

        if self.args.on_the_fly_feats:
            config = TorchAudioFbankConfig(
                sampling_rate=self.args.sampling_rate,
                n_mels=self.args.n_mels,
                n_fft=self.args.frame_length,
                hop_length=self.args.frame_shift,
            )
            train = SpeechSynthesisDataset(
                return_text=self.args.return_text,
                return_tokens=self.args.return_tokens,
                return_spk_ids=self.args.return_spk_ids,
                feature_input_strategy=OnTheFlyFeatures(TorchAudioFbank(config)),
                return_cuts=self.args.return_cuts,
                return_audio=self.args.return_audio,
            )

        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler.")
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                buffer_size=self.args.num_buckets * 2000,
                shuffle_buffer_size=self.args.num_buckets * 5000,
                drop_last=self.args.drop_last,
            )
        else:
            logging.info("Using SimpleCutSampler.")
            train_sampler = SimpleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
            )
        logging.info("About to create train dataloader")

        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        # 'seed' is derived from the current random state, which will have
        # previously been set in the main process.
        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)

        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
        )

        return train_dl

    def dev_dataloaders(self, cuts_valid: CutSet) -> DataLoader:
        logging.info("About to create dev dataset")
        if self.args.on_the_fly_feats:
            config = TorchAudioFbankConfig(
                sampling_rate=self.args.sampling_rate,
                n_mels=self.args.n_mels,
                n_fft=self.args.frame_length,
                hop_length=self.args.frame_shift,
            )
            validate = SpeechSynthesisDataset(
                return_text=self.args.return_text,
                return_tokens=self.args.return_tokens,
                return_spk_ids=self.args.return_spk_ids,
                feature_input_strategy=OnTheFlyFeatures(TorchAudioFbank(config)),
                return_cuts=self.args.return_cuts,
                return_audio=self.args.return_audio,
            )
        else:
            validate = SpeechSynthesisDataset(
                return_text=self.args.return_text,
                return_tokens=self.args.return_tokens,
                return_spk_ids=self.args.return_spk_ids,
                feature_input_strategy=eval(self.args.input_strategy)(),
                return_cuts=self.args.return_cuts,
                return_audio=self.args.return_audio,
            )
        dev_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        logging.info("About to create valid dataloader")
        dev_dl = DataLoader(
            validate,
            sampler=dev_sampler,
            batch_size=None,
            num_workers=2,
            persistent_workers=False,
        )

        return dev_dl

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        logging.info("About to create test dataset")
        if self.args.on_the_fly_feats:
            config = TorchAudioFbankConfig(
                sampling_rate=self.args.sampling_rate,
                n_mels=self.args.n_mels,
                n_fft=self.args.frame_length,
                hop_length=self.args.frame_shift,
            )
            test = SpeechSynthesisDataset(
                return_text=self.args.return_text,
                return_tokens=self.args.return_tokens,
                return_spk_ids=self.args.return_spk_ids,
                feature_input_strategy=OnTheFlyFeatures(TorchAudioFbank(config)),
                return_cuts=self.args.return_cuts,
                return_audio=self.args.return_audio,
                return_prompt=self.args.return_prompt,
            )
        else:
            test = SpeechSynthesisDataset(
                return_text=self.args.return_text,
                return_tokens=self.args.return_tokens,
                return_spk_ids=self.args.return_spk_ids,
                feature_input_strategy=eval(self.args.input_strategy)(),
                return_cuts=self.args.return_cuts,
                return_audio=self.args.return_audio,
                return_prompt=self.args.return_prompt,
            )

        if self.args.return_prompt:
            test_sampler = DynamicBucketingSampler(
                cuts,
                constraint=PromptTimeConstraint(max_duration=self.args.max_duration),
                shuffle=False,
            )
        else:
            test_sampler = DynamicBucketingSampler(
                cuts,
                max_duration=self.args.max_duration,
                shuffle=False,
            )
        logging.info("About to create test dataloader")
        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=test_sampler,
            num_workers=self.args.num_workers,
        )
        return test_dl

    @lru_cache()
    def train_all_shuf_cuts(self) -> CutSet:
        logging.info(
            "About to get the shuffled train-clean-100, \
            train-clean-360 and train-other-500 cuts"
        )
        return load_manifest_lazy(
            self.args.manifest_dir / "libritts_cuts_with_tokens_train-all-shuf.jsonl.gz"
        )

    @lru_cache()
    def dev_clean_cuts(self) -> CutSet:
        logging.info("About to get dev-clean cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "libritts_cuts_with_tokens_dev-clean.jsonl.gz"
        )

    @lru_cache()
    def test_clean_cuts(self) -> CutSet:
        logging.info("About to get test-clean cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "libritts_cuts_with_tokens_test-clean.jsonl.gz"
        )

    @lru_cache()
    def test_clean_prompt_cuts(self) -> CutSet:
        logging.info("About to get test-clean cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "libritts_cuts_with_prompts_test-clean.jsonl.gz"
        )

    @lru_cache()
    def librispeech_test_clean_prompt_cuts(self) -> CutSet:
        logging.info("About to get test-clean cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_with_prompts_test-clean.jsonl.gz"
        )


def validate_for_tts(cuts: CutSet) -> None:
    validate(cuts)
    for cut in cuts:
        assert (
            len(cut.supervisions) == 1
        ), "Only the Cuts with single supervision are supported."
