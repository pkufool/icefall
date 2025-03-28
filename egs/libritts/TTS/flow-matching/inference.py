#!/usr/bin/env python3
# Copyright         2024  Xiaomi Corp.        (authors: Wei Kang
#                                                       Han Zhu)
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
from pathlib import Path

import torch
import torch.nn as nn
from checkpoint import load_checkpoint
from lhotse.utils import fix_random_seed
from model import get_distill_model, get_model
from scipy.io.wavfile import write
from tokenizer import Tokenizer
from train import add_model_arguments, get_params
from tts_datamodule import TorchAudioFbank, TorchAudioFbankConfig, TtsDataModule
from utils import prepare_input, save_plot
from vocos import Vocos

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
)
from icefall.utils import AttributeDict, setup_logger, str2bool

LOG_EPS = math.log(1e-10)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=1000,
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
        default=10,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=False,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="flow-matching/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--feat-scale",
        type=float,
        default=0.1,
        help="The scale factor of fbank feature",
    )

    parser.add_argument(
        "--solver",
        type=str,
        default="dpm",
        choices=["dpm", "euler"],
        help="The type of ODE solver.",
    )

    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=0.8,
        help="The scale of classifier-free guidance during inference.",
    )

    parser.add_argument(
        "--num-step",
        type=int,
        default=10,
        help="The number of forward steps.",
    )

    parser.add_argument(
        "--duration",
        type=str,
        default="predict",
        choices=["real", "predict"],
        help="Where the duration comes from. ",
    )

    parser.add_argument(
        "--vocoder-path",
        type=str,
        default="model/huggingface/vocos-mel-24khz/",
        help="The vocos vocoder path, downloaded from huggingface.",
    )

    parser.add_argument(
        "--distill",
        type=str2bool,
        default=False,
        help="Whether it is a distilled TTS model.",
    )

    parser.add_argument(
        "--generate-dir",
        type=str,
        default="generated_wavs",
        help="Path name of the generated wavs",
    )

    parser.add_argument(
        "--text-with-prompt",
        type=str,
        help="""
        The input file with the following format:
        target_audio_name \t prompt_text \t prompt_audio_path \t text
        """,
    )

    parser.add_argument(
        "--vocoder-only",
        type=str2bool,
        default=False,
        help="Whether to use ground-truth features for vocoder resynthesizing.",
    )

    parser.add_argument(
        "--save-verbose",
        type=str2bool,
        default=False,
        help="Whether to save more results for analysis, specifically, save the"
        "ground-truth audios, prompts and plot figures of features.",
    )

    add_model_arguments(parser)

    return parser


def get_vocoder(vocos_local_path: str):
    vocos_local_path = "model/huggingface/vocos-mel-24khz/"
    vocoder = Vocos.from_hparams(f"{vocos_local_path}/config.yaml")
    state_dict = torch.load(
        f"{vocos_local_path}/pytorch_model.bin", weights_only=True, map_location="cpu"
    )
    vocoder.load_state_dict(state_dict)
    return vocoder


def decode_texts(
    params: AttributeDict, model: nn.Module, vocoder: nn.Module, tokenizer: Tokenizer
):
    """
    Run inference on the given prompt and text pairs.
    """
    total_rtf = []
    total_rtf_no_vocoder = []
    total_rtf_vocoder = []

    config = TorchAudioFbankConfig(
        sampling_rate=params.sampling_rate,
        n_mels=params.n_mels,
        n_fft=params.frame_length,
        hop_length=params.frame_shift,
    )
    feature_extractor = TorchAudioFbank(config)
    device = params.device

    with open(params.text_with_prompt, "r") as fr:
        lines = fr.readlines()
    with torch.inference_mode():
        for i, line in enumerate(lines):
            audio_name, prompt_text, prompt_audio, text = line.strip().split("\t")
            tokens = tokenizer.texts_to_token_ids([text])
            prompt_tokens = tokenizer.texts_to_token_ids([prompt_text])
            prompt_audio, sampling_rate = torchaudio.load(prompt_audio)
            assert sampling_rate == params.sampling_rate, (
                sampling_rate,
                params.sampling_rate,
            )
            prompt_features = feature_extractor.extract(
                prompt_audio, sampling_rate=sampling_rate
            ).to(device)
            prompt_features_lens = torch.tensor(
                [prompt_features.size(1)], device=device
            )
            start_t = dt.datetime.now()

            (
                pred_features,
                pred_features_lens,
                pred_prompt_features,
                pred_prompt_features_lens,
            ) = model.sample(
                tokens=tokens,
                prompt_tokens=prompt_tokens,
                prompt_features=prompt_features,
                prompt_features_lens=prompt_features_lens,
                duration=params.duration,
                solver=params.solver,
                num_step=params.num_step,
                guidance_scale=params.guidance_scale,
                distill=params.distill,
            )

            pred_features = (
                pred_features.permute(0, 2, 1) / params.feat_scale
            )  # (B, C, T)

            start_vocoder_t = dt.datetime.now()
            audios = vocoder.forward(prompt_features).squeeze(1).clamp(-1, 1)

            t = (dt.datetime.now() - start_t).total_seconds()
            t_no_vocoder = (start_vocoder_t - start_t).total_seconds()
            t_vocoder = (dt.datetime.now() - start_vocoder_t).total_seconds()

            audio = audios[0][
                : int(x1_lens[0] * params.frame_shift_ms / 1000 * params.sample_rate)
            ]

            rtf = t * params.sample_rate / (audio.shape[-1])
            rtf_no_vocoder = t_no_vocoder * params.sample_rate / (audio.shape[-1])
            rtf_vocoder = t_vocoder * params.sample_rate / (audio.shape[-1])

            print(f"[Batch: {i}] RTF: {rtf:.4f}")
            print(f"[Batch: {i}] RTF w/o vocoder: {rtf_no_vocoder:.4f}")
            print(f"[Batch: {i}] RTF vocoder: {rtf_vocoder:.4f}")

            total_rtf.append(rtf)
            total_rtf_no_vocoder.append(rtf_no_vocoder)
            total_rtf_vocoder.append(rtf_vocoder)
            audio = audio.cpu().numpy()

            write(f"{params.wav_dir}/{audio_name}.wav", params.sample_rate, audio)

    print(f"Average RTF: {np.mean(total_rtf[10:]):.4f}")
    print(f"Average RTF w/o vocoder: {np.mean(total_rtf_no_vocoder[10:]):.4f}")
    print(f"Average RTF vocoder: {np.mean(total_rtf_vocoder[10:]):.4f}")


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    vocoder: nn.Module,
    batch: dict,
):
    """
    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The text-to-feature neural model.
      vocoder:
        The vocoder neural model.
      batch:
        It is the return value from iterating
        `tts_datamodule.SpeechSynthesisDataset`. See its documentation
        for the format of the `batch`.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict.
    """
    device = next(model.parameters()).device

    cut_ids = [cut.id for cut in batch["cut"]]
    if not params.vocoder_only:
        """
        inputs:
            {
                'features': (B x NumFrames x NumFeatures) float tensor
                'features_lens': (B, ) int tensor
                'audio': (B x NumSamples) float tensor
                'audio_lens': (B, ) int tensor
                'token_ids': List[List[int]]  # when return_token_ids=True
                'prompt': Dict of above all. # when return_prompt=True
            }
        """
        inputs = prepare_input(
            params=params,
            batch=batch,
            device=device,
            return_token_ids=True,
            return_feature=True,
            return_prompt=True,
        )
        (
            pred_features,
            pred_features_lens,
            pred_prompt_features,
            pred_prompt_features_lens,
        ) = model.sample(
            tokens=inputs["token_ids"],
            prompt_tokens=inputs["prompt"]["token_ids"],
            prompt_features=inputs["prompt"]["features"],
            prompt_features_lens=inputs["prompt"]["features_lens"],
            features_lens=inputs["features_lens"],
            duration=params.duration,
            solver=params.solver,
            num_step=params.num_step,
            guidance_scale=params.guidance_scale,
            distill=params.distill,
        )
    else:
        inputs = prepare_input(
            params=params,
            batch=batch,
            device=device,
            return_feature=True,
        )
        pred_features = inputs["features"]
        pred_features_lens = inputs["features_lens"]

    pred_features = pred_features.permute(0, 2, 1) / params.feat_scale  # (B, C, T)

    for i in range(pred_features.shape[0]):
        audio = (
            vocoder.decode(pred_features[i][None, :, : pred_features_lens[i]])
            .squeeze(1)
            .clamp(-1, 1)
        )
        audio = audio[0].cpu().numpy()
        write(f"{params.wav_dir}/{cut_ids[i]}.wav", params.sample_rate, audio)

    if params.save_verbose:
        inputs = prepare_input(
            params=params,
            batch=batch,
            device=device,
            return_audio=True,
            return_prompt=True,
            return_feature=True,
        )
        gt_audio = inputs["audio"]
        gt_audio_len = inputs["audio_lens"]
        prompt_audio = inputs["prompt"]["audio"]
        prompt_audio_len = inputs["prompt"]["audio_lens"]
        features = inputs["features"].permute(0, 2, 1) / params.feat_scale  # (B, C, T)
        prompt_features = (
            inputs["prompt_features"].permute(0, 2, 1) / params.feat_scale
        )  # (B, C, T)
        pred_prompt_features = (
            pred_prompt_features.permute(0, 2, 1) / params.feat_scale
        )  # (B, C, T)

        for i in range(features.shape[0]):
            write(
                f"{params.wav_dir_gt}/{cut_ids[i]}.wav",
                params.sample_rate,
                gt_audio[i][: gt_audio_len[i]].cpu().numpy(),
            )
            write(
                f"{params.wav_dir_prompt}/{cut_ids[i]}.wav",
                params.sample_rate,
                prompt_audio[i][: prompt_audio_len[i]].cpu().numpy(),
            )

            save_plot(
                pred_features[i][:, : pred_features_lens[i]].cpu(),
                f"{params.fig_dir}/{cut_ids[i]}_predict.png",
            )
            save_plot(
                features[i][:, : features_lens[i]].cpu(),
                f"{params.fig_dir}/{cut_ids[i]}_real.png",
            )
            save_plot(
                pred_prompt_features[i]
                .transpose(0, 1)[:, : pred_prompt_features_lens[i]]
                .cpu(),
                f"{params.fig_dir}/{cut_ids[i]}_prompt_predict.png",
            )
            save_plot(
                prompt_features[i].transpose(0, 1)[:, : prompt_features_lens[i]].cpu(),
                f"{params.fig_dir}/{cut_ids[i]}_prompt_real.png",
            )


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    vocoder: nn.Module,
    test_set: str,
):
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The text-to-feature neural model.
      vocoder:
        The feature-to-speech neural model.
      test_set:
        The name of the test_set
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    with open(f"{params.wav_dir}/{test_set}.scp", "w", encoding="utf8") as f:
        for batch_idx, batch in enumerate(dl):
            texts = batch["text"]
            cut_ids = [cut.id for cut in batch["cut"]]

            decode_one_batch(
                params=params,
                model=model,
                vocoder=vocoder,
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
    TtsDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    params.res_dir = params.exp_dir / params.generate_dir

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}-step-{params.num_step}-scale-{params.guidance_scale}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}-step-{params.num_step}-scale-{params.guidance_scale}"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    params.device = device

    logging.info(f"Device: {device}")

    tokenizer = Tokenizer(token_type=params.token_type, token_file=params.token_file)
    params.vocab_size = tokenizer.vocab_size

    logging.info(params)
    fix_random_seed(666)

    logging.info("About to create model")
    if params.distill:
        model = get_distill_model(params)
    else:
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
            model.load_state_dict(
                average_checkpoints(filenames, device=device), strict=True
            )
        elif params.avg == 1:
            load_checkpoint(
                f"{params.exp_dir}/epoch-{params.epoch}.pt", model, strict=True
            )
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(
                average_checkpoints(filenames, device=device), strict=True
            )
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
                ),
                strict=True,
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
                ),
                strict=True,
            )

    model = model.to(device)
    model.eval()

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    vocoder = get_vocoder(params.vocoder_path)
    vocoder = vocoder.to(device)
    vocoder.eval()
    num_param = sum([p.numel() for p in vocoder.parameters()])
    logging.info(f"Number of vocoder parameters: {num_param}")

    params.wav_dir = f"{params.res_dir}/{params.suffix}"
    os.makedirs(params.wav_dir, exist_ok=True)

    if params.save_verbose:
        params.wav_dir_gt = f"{params.res_dir}/{params.suffix}_gt"
        os.makedirs(params.wav_dir_gt, exist_ok=True)

        params.wav_dir_prompt = f"{params.res_dir}/{params.suffix}_prompt"
        os.makedirs(params.wav_dir_prompt, exist_ok=True)

        params.fig_dir = f"{params.res_dir}/{params.suffix}_fig"
        os.makedirs(params.fig_dir, exist_ok=True)

    # we need cut ids to display recognition results.
    args.return_cuts = True
    libritts = TtsDataModule(args)

    test_cuts = libritts.librispeech_test_clean_prompt_cuts()

    test_dl = libritts.test_prompt_dataloaders(
        test_cuts, return_audio=params.save_verbose
    )

    test_sets = ["test"]
    test_dls = [test_dl]

    for test_set, test_dl in zip(test_sets, test_dls):
        decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            vocoder=vocoder,
            test_set=test_set,
        )

    logging.info("Done!")


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    main()
