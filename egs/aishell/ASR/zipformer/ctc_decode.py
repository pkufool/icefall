#!/usr/bin/env python3
#
# Copyright 2021-2022 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Liyong Guo,
#                                                 Quandong Wang,
#                                                 Zengwei Yao)
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

(1) ctc-greedy-search
./zipformer/ctc_decode.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --use-ctc 1 \
    --max-duration 600 \
    --decoding-method ctc-greedy-search

(2) ctc-decoding
./zipformer/ctc_decode.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --use-ctc 1 \
    --max-duration 600 \
    --decoding-method ctc-decoding

(3) 1best
./zipformer/ctc_decode.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --use-ctc 1 \
    --max-duration 600 \
    --hlg-scale 0.6 \
    --decoding-method 1best

(4) nbest
./zipformer/ctc_decode.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --use-ctc 1 \
    --max-duration 600 \
    --hlg-scale 0.6 \
    --decoding-method nbest

(5) nbest-rescoring
./zipformer/ctc_decode.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --use-ctc 1 \
    --max-duration 600 \
    --hlg-scale 0.6 \
    --nbest-scale 1.0 \
    --lm-dir data/lm \
    --decoding-method nbest-rescoring

(6) whole-lattice-rescoring
./zipformer/ctc_decode.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --use-ctc 1 \
    --max-duration 600 \
    --hlg-scale 0.6 \
    --nbest-scale 1.0 \
    --lm-dir data/lm \
    --decoding-method whole-lattice-rescoring

(7) attention-decoder-rescoring-no-ngram
./zipformer/ctc_decode.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --use-ctc 1 \
    --use-attention-decoder 1 \
    --max-duration 100 \
    --decoding-method attention-decoder-rescoring-no-ngram

(8) attention-decoder-rescoring-with-ngram
./zipformer/ctc_decode.py \
    --epoch 30 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --use-ctc 1 \
    --use-attention-decoder 1 \
    --max-duration 100 \
    --hlg-scale 0.6 \
    --nbest-scale 1.0 \
    --lm-dir data/lm \
    --decoding-method attention-decoder-rescoring-with-ngram
"""


import argparse
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import torch
import torch.nn as nn
from asr_datamodule import AishellAsrDataModule
from lhotse import set_caching_enabled
from lhotse.cut import Cut
from train import add_model_arguments, get_model, get_params

from icefall.context_graph import ContextGraph, ContextState
from icefall.ngram_lm import NgramLm, NgramLmStateCost
from icefall.lm_wrapper import LmScorer


from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.decode import (
    ctc_greedy_search,
    ctc_prefix_beam_search,
    ctc_prefix_beam_search_attention_decoder_rescoring,
    ctc_prefix_beam_search_shallow_fussion,
    get_lattice,
    one_best_decoding,
    rescore_with_attention_decoder_no_ngram,
)
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    get_texts,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)

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
        "--lang-dir",
        type=Path,
        default="data/lang_char",
        help="The lang dir containing word table and LG graph",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="ctc-decoding",
        help="""Decoding method.
        Supported values are:
        - (1) ctc-greedy-search. Use CTC greedy search. It uses a sentence piece
          model, i.e., lang_dir/bpe.model, to convert word pieces to words.
          It needs neither a lexicon nor an n-gram LM.
        - (2) ctc-decoding. Use CTC decoding. It uses a sentence piece
          model, i.e., lang_dir/bpe.model, to convert word pieces to words.
          It needs neither a lexicon nor an n-gram LM.
        - (3) attention-decoder-rescoring-no-ngram. Extract n paths from the decoding
          lattice, rescore them with the attention decoder.
        """,
    )

    parser.add_argument(
        "--num-paths",
        type=int,
        default=100,
        help="""Number of paths for n-best based decoding method.
        Used only when "method" is one of the following values:
        nbest, nbest-rescoring, and nbest-oracle
        """,
    )

    parser.add_argument(
        "--nbest-scale",
        type=float,
        default=1.0,
        help="""The scale to be applied to `lattice.scores`.
        It's needed if you use any kinds of n-best based rescoring.
        Used only when "method" is one of the following values:
        nbest, nbest-rescoring, and nbest-oracle
        A smaller value results in more unique paths.
        """,
    )

    parser.add_argument(
        "--skip-scoring",
        type=str2bool,
        default=False,
        help="""Skip scoring, but still save the ASR output (for eval sets).""",
    )

    parser.add_argument(
        "--lm-type",
        type=str,
        default="rnn",
        help="Type of NN lm",
        choices=["rnn", "transformer"],
    )

    parser.add_argument(
        "--lm-scale",
        type=float,
        default=0.3,
        help="""The scale of the neural network LM
        Used only when `--use-shallow-fusion` is set to True.
        """,
    )

    add_model_arguments(parser)

    return parser


def get_decoding_params() -> AttributeDict:
    """Parameters for decoding."""
    params = AttributeDict(
        {
            "frame_shift_ms": 10,
            "search_beam": 20,  # for k2 fsa composition
            "output_beam": 8,  # for k2 fsa composition
            "beam": 4,  # for prefix-beam-search
            "min_active_states": 30,
            "max_active_states": 10000,
            "use_double_scores": True,
        }
    )
    return params


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    lexicon: Lexicon,
    batch: dict,
    H: Optional[k2.Fsa],
    LM: Optional[LmScorer] = None,
) -> Dict[str, List[List[str]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:
    - key: It indicates the setting used for decoding. For example,
           if no rescoring is used, the key is the string `no_rescore`.
           If LM rescoring is used, the key is the string `lm_scale_xxx`,
           where `xxx` is the value of `lm_scale`. An example key is
           `lm_scale_0.7`
    - value: It contains the decoding result. `len(value)` equals to
             batch size. `value[i]` is the decoding result for the i-th
             utterance in the given batch.

    Args:
      params:
        It's the return value of :func:`get_params`.

        - params.decoding_method is "1best", it uses 1best decoding without LM rescoring.
        - params.decoding_method is "nbest", it uses nbest decoding without LM rescoring.
        - params.decoding_method is "nbest-rescoring", it uses nbest LM rescoring.
        - params.decoding_method is "whole-lattice-rescoring", it uses whole lattice LM
          rescoring.

      model:
        The neural model.
      HLG:
        The decoding graph. Used only when params.decoding_method is NOT ctc-decoding.
      H:
        The ctc topo. Used only when params.decoding_method is ctc-decoding.
      bpe_model:
        The BPE model. Used only when params.decoding_method is ctc-decoding.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
      word_table:
        The word symbol table.
      G:
        An LM. It is not None when params.decoding_method is "nbest-rescoring"
        or "whole-lattice-rescoring". In general, the G in HLG
        is a 3-gram LM, while this G is a 4-gram LM.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict. Note: If it decodes to nothing, then return None.
    """
    # TODO
    device = next(model.parameters()).device
    feature = batch["inputs"]
    assert feature.ndim == 3
    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    if params.causal:
        # this seems to cause insertions at the end of the utterance if used with zipformer.
        pad_len = 30
        feature_lens += pad_len
        feature = torch.nn.functional.pad(
            feature,
            pad=(0, 0, 0, pad_len),
            value=LOG_EPS,
        )

    encoder_out, encoder_out_lens = model.forward_encoder(feature, feature_lens)
    ctc_output = model.ctc_output(encoder_out)  # (N, T, C)

    batch_size = encoder_out.size(0)

    if params.decoding_method == "ctc-greedy-search":
        hyp_tokens = ctc_greedy_search(ctc_output, encoder_out_lens)
        hyps = []
        for i in range(batch_size):
            hyps.append([lexicon.token_table[idx] for idx in hyp_tokens[i]])
        key = "ctc-greedy-search"
        return {key: hyps}

    if params.decoding_method == "prefix-beam-search":
        hyp_tokens = ctc_prefix_beam_search(
            ctc_output=ctc_output, encoder_out_lens=encoder_out_lens
        )
        hyps = []
        for i in range(batch_size):
            hyps.append([lexicon.token_table[idx] for idx in hyp_tokens[i]])
        key = "prefix-beam-search"
        return {key: hyps}

    if params.decoding_method == "ctc-prefix-beam-search-attention-decoder-rescoring":
        best_path_dict = ctc_prefix_beam_search_attention_decoder_rescoring(
            ctc_output=ctc_output,
            attention_decoder=model.attention_decoder,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
        )
        ans = dict()
        for a_scale_str, hyp_tokens in best_path_dict.items():
            hyps = []
            for i in range(batch_size):
                hyps.append([lexicon.token_table[idx] for idx in hyp_tokens[i]])
            ans[a_scale_str] = hyps
        return ans

    if params.decoding_method == "ctc-prefix-beam-search-shallow-fussion":
        hyp_tokens = ctc_prefix_beam_search_shallow_fussion(
            ctc_output=ctc_output,
            encoder_out_lens=encoder_out_lens,
            LM=LM,
        )
        hyps = []
        for i in range(batch_size):
            hyps.append([lexicon.token_table[idx] for idx in hyp_tokens[i]])
        key = "prefix-beam-search-shallow-fussion"
        return {key: hyps}

    supervision_segments = torch.stack(
        (
            supervisions["sequence_idx"],
            torch.div(
                supervisions["start_frame"],
                params.subsampling_factor,
                rounding_mode="floor",
            ),
            torch.div(
                supervisions["num_frames"],
                params.subsampling_factor,
                rounding_mode="floor",
            ),
        ),
        1,
    ).to(torch.int32)

    assert H is not None
    decoding_graph = H
    lattice = get_lattice(
        nnet_output=ctc_output,
        decoding_graph=decoding_graph,
        supervision_segments=supervision_segments,
        search_beam=params.search_beam,
        output_beam=params.output_beam,
        min_active_states=params.min_active_states,
        max_active_states=params.max_active_states,
        subsampling_factor=params.subsampling_factor,
    )

    if params.decoding_method == "ctc-decoding":
        best_path = one_best_decoding(
            lattice=lattice, use_double_scores=params.use_double_scores
        )
        # Note: `best_path.aux_labels` contains token IDs, not word IDs
        # since we are using H, not HLG here.
        #
        # token_ids is a lit-of-list of IDs
        hyp_tokens = get_texts(best_path)
        hyps = []
        for i in range(encoder_out.size(0)):
            hyps.append([lexicon.token_table[idx] for idx in hyp_tokens[i]])
        key = "ctc-decoding"
        return {key: hyps}  # note: returns words

    if params.decoding_method == "attention-decoder-rescoring-no-ngram":
        best_path_dict = rescore_with_attention_decoder_no_ngram(
            lattice=lattice,
            num_paths=params.num_paths,
            attention_decoder=model.attention_decoder,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            nbest_scale=params.nbest_scale,
        )
        ans = dict()
        for a_scale_str, best_path in best_path_dict.items():
            # token_ids is a lit-of-list of IDs
            hyps = []
            hyp_tokens = get_texts(best_path)
            for i in range(encoder_out.size(0)):
                hyps.append([lexicon.token_table[idx] for idx in hyp_tokens[i]])
            ans[a_scale_str] = hyps
        return ans
    else:
        assert False, f"Unsupported decoding method: {params.decoding_method}"


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    lexicon: Lexicon,
    H: Optional[k2.Fsa] = None,
    LM: Optional[LmScorer] = None,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      HLG:
        The decoding graph. Used only when params.decoding_method is NOT ctc-decoding.
      H:
        The ctc topo. Used only when params.decoding_method is ctc-decoding.
      bpe_model:
        The BPE model. Used only when params.decoding_method is ctc-decoding.
      word_table:
        It is the word symbol table.
      G:
        An LM. It is not None when params.decoding_method is "nbest-rescoring"
        or "whole-lattice-rescoring". In general, the G in HLG
        is a 3-gram LM, while this G is a 4-gram LM.
    Returns:
      Return a dict, whose key may be "no-rescore" if no LM rescoring
      is used, or it may be "lm_scale_0.7" if LM rescoring is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        texts = [list("".join(text.split())) for text in texts]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            batch=batch,
            lexicon=lexicon,
            H=H,
            LM=LM,
        )

        for name, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)
            for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
                this_batch.append((cut_id, ref_text, hyp_words))

            results[name].extend(this_batch)

        num_cuts += len(texts)

        if batch_idx % 100 == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results


def save_asr_output(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    """
    Save text produced by ASR.
    """
    for key, results in results_dict.items():

        recogs_filename = (
            params.res_dir / f"recogs-{test_set_name}-{key}-{params.suffix}.txt"
        )

        results = sorted(results)
        store_transcripts(filename=recogs_filename, texts=results, char_level=True)

        logging.info(f"The transcripts are stored in {recogs_filename}")


def save_wer_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    if params.decoding_method == "attention-decoder-rescoring-no-ngram":
        # Set it to False since there are too many logs.
        enable_log = False
    else:
        enable_log = True

    test_set_wers = dict()
    for key, results in results_dict.items():
        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = (
            params.res_dir / f"errs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        with open(errs_filename, "w", encoding="utf8") as fd:
            wer = write_error_stats(
                fd,
                f"{test_set_name}-{key}",
                results,
                enable_log=enable_log,
                compute_CER=True,
            )
            test_set_wers[key] = wer

        logging.info(f"Wrote detailed error stats to {errs_filename}")

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])

    wer_filename = (
        params.res_dir / f"wer-summary-{test_set_name}-{key}-{params.suffix}.txt"
    )

    with open(wer_filename, "w", encoding="utf8") as fd:
        print("settings\tWER", file=fd)
        for key, val in test_set_wers:
            print(f"{key}\t{val}", file=fd)

    s = f"\nFor {test_set_name}, WER of different settings are:\n"
    note = f"\tbest for {test_set_name}"
    for key, val in test_set_wers:
        s += f"{key}\t{val}{note}\n"
        note = ""
    logging.info(s)


@torch.no_grad()
def main():
    parser = get_parser()
    AishellAsrDataModule.add_arguments(parser)
    LmScorer.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.lang_dir = Path(args.lang_dir)

    params = get_params()
    # add decoding params
    params.update(get_decoding_params())
    params.update(vars(args))

    # enable AudioCache
    set_caching_enabled(True)  # lhotse

    assert params.decoding_method in (
        "ctc-greedy-search",
        "prefix-beam-search",
        "ctc-prefix-beam-search-attention-decoder-rescoring",
        "ctc-prefix-beam-search-shallow-fussion",
        "ctc-decoding",
        "attention-decoder-rescoring-no-ngram",
    )
    params.res_dir = params.exp_dir / params.decoding_method

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    if params.causal:
        assert (
            "," not in params.chunk_size
        ), "chunk_size should be one value in decoding."
        assert (
            "," not in params.left_context_frames
        ), "left_context_frames should be one value in decoding."
        params.suffix += f"_chunk-{params.chunk_size}"
        params.suffix += f"_left-context-{params.left_context_frames}"

    if "prefix-beam-search" in params.decoding_method:
        params.suffix += f"_beam-{params.beam}"
        if params.decoding_method == "ctc-prefix-beam-search-shallow-fussion":
            params.suffix += f"_lm-scale-{params.lm_scale}"

    if params.use_averaged_model:
        params.suffix += "_use-averaged-model"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")
    logging.info(params)

    lexicon = Lexicon(params.lang_dir)
    max_token_id = max(lexicon.tokens)
    num_classes = max_token_id + 1  # +1 for the blank

    params.vocab_size = num_classes
    # <blk> and <unk> are defined in local/train_bpe_model.py
    params.blank_id = 0
    params.eos_id = 1
    params.sos_id = 1

    if params.decoding_method in [
        "ctc-decoding",
        "attention-decoder-rescoring-no-ngram",
    ]:
        H = k2.ctc_topo(
            max_token=max_token_id,
            modified=True,
            device=device,
        )
    else:
        H = None

    # only load the neural network LM if required
    if params.decoding_method == "ctc-prefix-beam-search-shallow-fussion":
        LM = LmScorer(
            lm_type=params.lm_type,
            params=params,
            device=device,
            lm_scale=params.lm_scale,
        )
        LM.to(device)
        LM.eval()
    else:
        LM = None

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

    model.to(device)
    model.eval()

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    aishell = AishellAsrDataModule(args)

    def remove_short_utt(c: Cut):
        T = ((c.num_frames - 7) // 2 + 1) // 2
        if T <= 0:
            logging.warning(
                f"Exclude cut with ID {c.id} from decoding, num_frames : {c.num_frames}."
            )
        return T > 0

    dev_cuts = aishell.valid_cuts()
    dev_cuts = dev_cuts.filter(remove_short_utt)
    dev_dl = aishell.valid_dataloaders(dev_cuts)

    test_cuts = aishell.test_cuts()
    test_cuts = test_cuts.filter(remove_short_utt)
    test_dl = aishell.test_dataloaders(test_cuts)

    test_sets = ["dev", "test"]
    test_dls = [dev_dl, test_dl]

    for test_set, test_dl in zip(test_sets, test_dls):
        results_dict = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            H=H,
            lexicon=lexicon,
            LM=LM,
        )

        save_asr_output(
            params=params,
            test_set_name=test_set,
            results_dict=results_dict,
        )

        if not params.skip_scoring:
            save_wer_results(
                params=params,
                test_set_name=test_set,
                results_dict=results_dict,
            )

    logging.info("Done!")


if __name__ == "__main__":
    main()
