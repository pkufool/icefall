#!/usr/bin/env python3
#
# Combined CTC and RNN-T decoding script
# Supports a minimal set of decoding methods:
#   - rnnt-greedy-search
#   - rnnt-modified-beam-search
#   - ctc-greedy-search
#   - ctc-prefix-beam-search
#

import argparse
import logging
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from beam_search import greedy_search_batch, modified_beam_search
from icefall.decode import ctc_greedy_search, ctc_prefix_beam_search
from icefall.utils import (
    AttributeDict,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)
from lhotse.cut import CutSet
from ssentencepiece import Ssentencepiece
from train_bpe import add_model_arguments, get_model, get_params
from asr_datamodule import AsrDataModule

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)

LOG_EPS = math.log(1e-10)

conversational_filler = [
    "UH",
    "UHH",
    "UM",
    "EH",
    "MM",
    "HM",
    "AH",
    "HUH",
    "HA",
    "ER",
    "OOF",
    "HEE",
    "ACH",
    "EEE",
    "EW",
]
unk_tags = ["<UNK>", "<unk>"]
gigaspeech_punctuations = [
    "<COMMA>",
    "<PERIOD>",
    "<QUESTIONMARK>",
    "<EXCLAMATIONPOINT>",
]
gigaspeech_garbage_utterance_tags = ["<SIL>", "<NOISE>", "<MUSIC>", "<OTHER>"]
non_scoring_words = (
    conversational_filler
    + unk_tags
    + gigaspeech_punctuations
    + gigaspeech_garbage_utterance_tags
)


def asr_text_post_processing(text: str) -> str:
    # 1. convert to uppercase
    text = text.upper()

    # 2. remove hyphen
    #   "E-COMMERCE" -> "E COMMERCE", "STATE-OF-THE-ART" -> "STATE OF THE ART"
    text = text.replace("-", " ")

    # 3. remove non-scoring words from evaluation
    remaining_words = []
    for word in text.split():
        if word in non_scoring_words:
            continue
        remaining_words.append(word)

    return " ".join(remaining_words).lower()


ALLOWED_METHODS = {
    "rnnt-greedy-search",
    "rnnt-modified-beam-search",
    "ctc-greedy-search",
    "ctc-prefix-beam-search",
}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--avg", type=int, default=15)
    parser.add_argument("--use-averaged-model", type=str2bool, default=True)

    parser.add_argument("--exp-dir", type=str, default="zipformer/exp")
    parser.add_argument("--bpe-model", type=str, default="data/lang_bpe_500/bpe.model")

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="rnnt-greedy-search",
        choices=sorted(ALLOWED_METHODS),
        help="Choose between rnnt or ctc decoding variants",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="Beam size used by rnnt modified beam search and ctc prefix beam search",
    )

    parser.add_argument(
        "--max-sym-per-frame",
        type=int,
        default=1,
        help="Maximum symbols per frame for rnnt greedy search",
    )
    parser.add_argument(
        "--search-avg",
        type=str2bool,
        default=True,
        help="Whether to search averaged model.",
    )

    parser.add_argument(
        "--test-manifest",
        type=str,
        action="append",
        default=[],
        help="Test manifest in the format 'test_name,manifest_path'. "
        "Can be specified multiple times for multiple test sets. "
        "Example: --test-manifest test-clean,data/fbank/librispeech_cuts_test-clean.jsonl.gz "
        "--test-manifest test-other,data/fbank/librispeech_cuts_test-other.jsonl.gz",
    )
    add_model_arguments(parser)
    return parser


def _load_checkpoint(
    params: AttributeDict, model: nn.Module, device: torch.device
) -> None:
    """Load checkpoints with or without averaging."""
    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for --iter {params.iter}, --avg {params.avg}"
                )
            if len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
        elif params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = [
                f"{params.exp_dir}/epoch-{i}.pt"
                for i in range(start, params.epoch + 1)
                if i >= 1
            ]
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
                    f"No checkpoints found for --iter {params.iter}, --avg {params.avg}"
                )
            if len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints "
                f"from {filename_start} (excluded) to {filename_end}"
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
            assert params.avg > 0
            start = params.epoch - params.avg
            assert start >= 1
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from {start} (excluded) to {params.epoch}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )


def tokenize_by_cjk_char(line: str) -> List[str]:
    pattern = re.compile(
        r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
    )
    chars = pattern.split(line.strip())
    words: List[str] = []
    for w in chars:
        if w.strip():
            words.extend(w.split())
    return words


def decode_batch_rnnt(
    params: AttributeDict,
    model: nn.Module,
    sp: Ssentencepiece,
    batch: dict,
) -> Dict[str, List[List[str]]]:
    device = next(model.parameters()).device
    features = batch["inputs"]
    # at entry, features is (N, T, C)
    assert features.ndim == 3
    features = features.to(device)
    feature_lens = batch["supervisions"]["num_frames"].to(device)

    if params.causal:
        pad_len = 30
        feature_lens = feature_lens + pad_len
        features = torch.nn.functional.pad(
            features, pad=(0, 0, 0, pad_len), value=LOG_EPS
        )

    encoder_out, encoder_out_lens = model.forward_encoder(features, feature_lens)[0:2]

    if params.decoding_method == "rnnt-greedy-search":
        hyp_tokens = greedy_search_batch(
            model=model, encoder_out=encoder_out, encoder_out_lens=encoder_out_lens
        )
        hyps = [sp.decode(hyp).split() for hyp in hyp_tokens]
        return {"rnnt-greedy-search": hyps}

    assert params.decoding_method == "rnnt-modified-beam-search"
    hyp_tokens = modified_beam_search(
        model=model,
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
        beam=params.beam_size,
    )
    hyps = [sp.decode(hyp).split() for hyp in hyp_tokens]
    return {f"rnnt-modified-beam-search_beam-{params.beam_size}": hyps}


def decode_batch_ctc(
    params: AttributeDict,
    model: nn.Module,
    sp: Ssentencepiece,
    batch: dict,
) -> Dict[str, List[List[str]]]:
    device = params.device
    features = batch["inputs"]
    # at entry, features is (N, T, C)
    assert features.ndim == 3
    features = features.to(device)
    feature_lens = batch["supervisions"]["num_frames"].to(device)

    if params.causal:
        pad_len = 30
        feature_lens = feature_lens + pad_len
        features = torch.nn.functional.pad(
            features, pad=(0, 0, 0, pad_len), value=LOG_EPS
        )

    encoder_out, encoder_out_lens = model.forward_encoder(features, feature_lens)[0:2]
    ctc_output = model.ctc_output(encoder_out)

    if params.decoding_method == "ctc-greedy-search":
        token_ids = ctc_greedy_search(
            ctc_output=ctc_output,
            encoder_out_lens=encoder_out_lens,
            blank_id=params.blank_id,
        )
        hyps = [sp.decode(t).split() for t in token_ids]
        return {"ctc-greedy-search": hyps}

    assert params.decoding_method == "ctc-prefix-beam-search"
    token_ids = ctc_prefix_beam_search(
        ctc_output=ctc_output,
        encoder_out_lens=encoder_out_lens,
        beam=params.beam_size,
        blank_id=params.blank_id,
    )
    hyps = [sp.decode(t).split() for t in token_ids]
    return {f"ctc-prefix-beam-search_beam-{params.beam_size}": hyps}


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: Ssentencepiece,
    decode_fn,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    num_cuts = 0
    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    results = defaultdict(list)
    log_interval = 20

    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        cuts = batch["supervisions"].get("cut", None)
        if cuts is not None:
            cut_ids = [c.id for c in cuts]
        else:
            cut_ids = list(range(len(texts)))

        hyps_dict = decode_fn(params=params, model=model, sp=sp, batch=batch)

        for name, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)
            for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
                ref_words = ref_text.strip()
                hyp_words = " ".join(hyp_words).strip()

                ref_words = asr_text_post_processing(ref_words)
                hyp_words = asr_text_post_processing(hyp_words)

                ref_words = re.sub(r"[,\.?!\"，。？！“”：:、<>《》\[\]{}【】;；]", "", ref_words)
                hyp_words = re.sub(r"[,\.?!\"，。？！“”：:、<>《》\[\]{}【】;；]", "", hyp_words)
                ref_words = tokenize_by_cjk_char(ref_words.lower())
                hyp_words = tokenize_by_cjk_char(hyp_words.lower())

                this_batch.append((cut_id, ref_words, hyp_words))
            results[name].extend(this_batch)

        num_cuts += len(texts)
        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"
            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")

    return results


def save_asr_output(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    for key, results in results_dict.items():
        recogs_filename = params.res_dir / f"recogs-{test_set_name}-{params.suffix}.txt"
        results = sorted(results)
        store_transcripts(filename=recogs_filename, texts=results)
        logging.info(f"The transcripts are stored in {recogs_filename}")


def save_wer_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    test_set_wers = dict()
    for key, results in results_dict.items():
        errs_filename = params.res_dir / f"errs-{test_set_name}-{params.suffix}.txt"
        with open(errs_filename, "w", encoding="utf8") as fd:
            wer = write_error_stats(
                fd, f"{test_set_name}-{key}", results, enable_log=True
            )
            test_set_wers[key] = wer
        logging.info(f"Wrote detailed error stats to {errs_filename}")

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    wer_filename = params.res_dir / f"wer-summary-{test_set_name}-{params.suffix}.txt"
    with open(wer_filename, "w", encoding="utf8") as fd:
        print("settings\tWER", file=fd)
        for key, val in test_set_wers:
            print(f"{key}\t{val}", file=fd)

    summary = f"\nFor {test_set_name}, WER of different settings are:\n"
    note = f"\tbest for {test_set_name}"
    for key, val in test_set_wers:
        summary += f"{key}\t{val}{note}\n"
        note = ""
    logging.info(summary)


def load_manifest_lazy(manifest_path: str) -> CutSet:
    """Load a manifest lazily from a path."""
    manifest_path = Path(manifest_path)
    if manifest_path.suffix == ".gz":
        return CutSet.from_jsonl_lazy(manifest_path)
    elif manifest_path.suffix in (".json", ".yaml"):
        return CutSet.from_file(manifest_path)
    else:
        return CutSet.from_jsonl_lazy(manifest_path)


@torch.no_grad()
def main():
    parser = get_parser()
    AsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    assert params.decoding_method in ALLOWED_METHODS

    params.res_dir = params.exp_dir / params.decoding_method
    if params.iter > 0:
        params.suffix = f"iter-{params.iter}_avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}_avg-{params.avg}"

    if params.causal:
        assert (
            "," not in params.chunk_size
        ), "chunk_size should be one value in decoding."
        assert (
            "," not in params.left_context_frames
        ), "left_context_frames should be one value in decoding."
        params.suffix += (
            f"_chunk-{params.chunk_size}_left-context-{params.left_context_frames}"
        )

    if "beam" in params.decoding_method:
        params.suffix += f"_beam-size-{params.beam_size}"

    if params.use_averaged_model:
        params.suffix += "_use-averaged-model"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )
    params.device = device
    logging.info(f"Device: {device}")

    sp = Ssentencepiece(params.bpe_model)
    params.blank_id = sp.piece_to_id("<blk>")
    params.sos_id = params.eos_id = sp.piece_to_id("<sos>")
    params.vocab_size = sp.vocab_size()

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)
    _load_checkpoint(params, model, device)
    model.to(device)
    model.eval()

    # Parse --test-manifest arguments: each is "test_name,manifest_path"
    test_sets = dict()
    for item in args.test_manifest:
        key, path = item.split(",", 1)
        test_sets[key] = path

    if not test_sets:
        logging.warning("No test manifests specified via --test-manifest. Nothing to decode.")
        return

    decode_fn = (
        decode_batch_ctc
        if params.decoding_method.startswith("ctc-")
        else decode_batch_rnnt
    )

    asr_datamodule = AsrDataModule(args)

    def remove_short_utt(c):
        T = ((c.num_frames - 7) // 2 + 1) // 2
        if T <= 0:
            logging.warning(
                f"Exclude cut with ID {c.id} from decoding, num_frames : {c.num_frames}."
            )
        return T > 0

    for test_set_name, manifest_path in test_sets.items():
        logging.info(f"Decoding test set: {test_set_name} from {manifest_path}")
        cuts = load_manifest_lazy(manifest_path)
        cuts = cuts.filter(remove_short_utt)
        test_dl = asr_datamodule.test_dataloaders(cuts)

        results_dict = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            sp=sp,
            decode_fn=decode_fn,
        )

        save_asr_output(
            params=params, test_set_name=test_set_name, results_dict=results_dict
        )
        save_wer_results(
            params=params, test_set_name=test_set_name, results_dict=results_dict
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
