#!/usr/bin/env python3
#
# Copyright 2021-2024 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Zengwei Yao,
#                                                 Xiaoyu Yang,
#                                                 Wei Kang)
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
The script gets forced-alignments based on the modified_beam_search decoding method.
The token-level alignments are saved to the new cuts manifests.

It loads a checkpoint and uses it to get the forced-alignments.


You can download a pretrained checkpoint from:


Usage of this script:

local/compute_alignment.py \
    --checkpoint ./exp/jit_script.pt \
    --bpe-model data/lang_bpe_256_punc/bpe.model \
    --max-duration 300 \
    --beam-size 4 \
    --manifests-in ljspeech_cuts_all.jsonl.gz \
    --manifests-out ljspeech_cuts_all_ali.jsonl.gz
"""
import argparse
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import AsrDataModule
from lhotse import CutSet, load_manifest_lazy
from lhotse.cut import Cut
from lhotse.serialization import SequentialJsonlWriter
from lhotse.supervision import AlignmentItem

from icefall.utils import AttributeDict, convert_timestamp, parse_timestamp


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint. "
        "The checkpoint is assumed to be a exported jit.script model.",
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
        required=True,
        help="The path to the input manifests.",
    )

    parser.add_argument(
        "--manifests-out",
        type=str,
        required=True,
        help="The path to the output manifests.",
    )

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing decoding parameters."""
    params = AttributeDict(
        {
            "subsampling_factor": 4,
            "frame_shift_ms": 10,
            "beam_size": 4,
        }
    )
    return params


@dataclass
class Hypothesis:
    # The predicted tokens so far.
    # Newly predicted tokens are appended to `ys`.
    ys: List[int]

    # The log prob of ys.
    # It contains only one entry.
    log_prob: torch.Tensor

    # timestamp[i] is the frame index after subsampling
    # on which ys[i] is decoded
    timestamp: List[int] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Return a string representation of self.ys"""
        return "_".join(map(str, self.ys))


class HypothesisList(object):
    def __init__(self, data: Optional[Dict[str, Hypothesis]] = None) -> None:
        """
        Args:
          data:
            A dict of Hypotheses. Its key is its `value.key`.
        """
        if data is None:
            self._data = {}
        else:
            self._data = data

    @property
    def data(self) -> Dict[str, Hypothesis]:
        return self._data

    def add(self, hyp: Hypothesis) -> None:
        """Add a Hypothesis to `self`.

        If `hyp` already exists in `self`, its probability is updated using
        `log-sum-exp` with the existed one.

        Args:
          hyp:
            The hypothesis to be added.
        """
        key = hyp.key
        if key in self:
            old_hyp = self._data[key]  # shallow copy
            torch.logaddexp(old_hyp.log_prob, hyp.log_prob, out=old_hyp.log_prob)
        else:
            self._data[key] = hyp

    def get_most_probable(self, length_norm: bool = False) -> Hypothesis:
        """Get the most probable hypothesis, i.e., the one with
        the largest `log_prob`.

        Args:
          length_norm:
            If True, the `log_prob` of a hypothesis is normalized by the
            number of tokens in it.
        Returns:
          Return the hypothesis that has the largest `log_prob`.
        """
        if length_norm:
            return max(self._data.values(), key=lambda hyp: hyp.log_prob / len(hyp.ys))
        else:
            return max(self._data.values(), key=lambda hyp: hyp.log_prob)

    def topk(self, k: int, length_norm: bool = False) -> "HypothesisList":
        """Return the top-k hypothesis.

        Args:
          length_norm:
            If True, the `log_prob` of a hypothesis is normalized by the
            number of tokens in it.
        """
        hyps = list(self._data.items())

        if length_norm:
            hyps = sorted(
                hyps, key=lambda h: h[1].log_prob / len(h[1].ys), reverse=True
            )[:k]
        else:
            hyps = sorted(hyps, key=lambda h: h[1].log_prob, reverse=True)[:k]

        ans = HypothesisList(dict(hyps))
        return ans

    def __contains__(self, key: str):
        return key in self._data

    def __iter__(self):
        return iter(self._data.values())

    def __len__(self) -> int:
        return len(self._data)

    def __str__(self) -> str:
        s = []
        for key in self:
            s.append(key)
        return ", ".join(s)


def get_hyps_shape(hyps: List[HypothesisList]) -> k2.RaggedShape:
    """Return a ragged shape with axes [utt][num_hyps].

    Args:
      hyps:
        len(hyps) == batch_size. It contains the current hypothesis for
        each utterance in the batch.
    Returns:
      Return a ragged shape with 2 axes [utt][num_hyps]. Note that
      the shape is on CPU.
    """
    num_hyps = [len(h) for h in hyps]

    # torch.cumsum() is inclusive sum, so we put a 0 at the beginning
    # to get exclusive sum later.
    num_hyps.insert(0, 0)

    num_hyps = torch.tensor(num_hyps)
    row_splits = torch.cumsum(num_hyps, dim=0, dtype=torch.int32)
    ans = k2.ragged.create_ragged_shape2(
        row_splits=row_splits, cached_tot_size=row_splits[-1].item()
    )
    return ans


# The force alignment problem can be formulated as finding
# a path in a rectangular lattice, where the path starts
# from the lower left corner and ends at the upper right
# corner. The horizontal axis of the lattice is `t` (representing
# acoustic frame indexes) and the vertical axis is `u` (representing
# BPE tokens of the transcript).
#
# The notations `t` and `u` are from the paper
# https://arxiv.org/pdf/1211.3711.pdf
#
# Beam search is used to find the path with the highest log probabilities.
#
# It assumes the maximum number of symbols that can be
# emitted per frame is 1.


def batch_force_alignment(
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    ys_list: List[List[int]],
    beam_size: int = 4,
) -> List[int]:
    """Compute the force alignment of a batch of utterances given their transcripts
    in BPE tokens and the corresponding acoustic output from the encoder.

    Caution:
      This function is modified from `modified_beam_search` in beam_search.py.
      We assume that the maximum number of sybmols per frame is 1.

    Args:
      model:
        The transducer model.
      encoder_out:
        A tensor of shape (N, T, C).
      encoder_out_lens:
        A 1-D tensor of shape (N,), containing number of valid frames in
        encoder_out before padding.
      ys_list:
        A list of BPE token IDs list. We require that for each utterance i,
        len(ys_list[i]) <= encoder_out_lens[i].
      beam_size:
        Size of the beam used in beam search.

    Returns:
      Return a list of frame indexes list for each utterance i,
      where len(ans[i]) == len(ys_list[i]).
    """
    assert encoder_out.ndim == 3, encoder_out.ndim
    assert encoder_out.size(0) == len(ys_list), (encoder_out.size(0), len(ys_list))
    assert encoder_out.size(0) > 0, encoder_out.size(0)

    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size
    device = next(model.parameters()).device

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    sorted_indices = packed_encoder_out.sorted_indices.tolist()
    encoder_out_lens = encoder_out_lens.tolist()
    ys_lens = [len(ys) for ys in ys_list]
    sorted_encoder_out_lens = [encoder_out_lens[i] for i in sorted_indices]
    sorted_ys_lens = [ys_lens[i] for i in sorted_indices]
    sorted_ys_list = [ys_list[i] for i in sorted_indices]

    B = [HypothesisList() for _ in range(N)]
    for i in range(N):
        B[i].add(
            Hypothesis(
                ys=[blank_id] * context_size,
                log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                timestamp=[],
            )
        )

    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)

    offset = 0
    finalized_B = []
    for t, batch_size in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        # current_encoder_out's shape is (batch_size, 1, 1, encoder_out_dim)
        offset = end

        finalized_B = B[batch_size:] + finalized_B
        B = B[:batch_size]
        sorted_encoder_out_lens = sorted_encoder_out_lens[:batch_size]
        sorted_ys_lens = sorted_ys_lens[:batch_size]

        hyps_shape = get_hyps_shape(B).to(device)

        A = [list(b) for b in B]
        B = [HypothesisList() for _ in range(batch_size)]

        ys_log_probs = torch.cat(
            [hyp.log_prob.reshape(1, 1) for hyps in A for hyp in hyps]
        )  # (num_hyps, 1)

        decoder_input = torch.tensor(
            [hyp.ys[-context_size:] for hyps in A for hyp in hyps],
            device=device,
            dtype=torch.int64,
        )  # (num_hyps, context_size)

        decoder_out = model.decoder(decoder_input, need_pad=False).unsqueeze(1)
        decoder_out = model.joiner.decoder_proj(decoder_out)
        # decoder_out is of shape (num_hyps, 1, 1, joiner_dim)

        # Note: For torch 1.7.1 and below, it requires a torch.int64 tensor
        # as index, so we use `to(torch.int64)` below.
        current_encoder_out = torch.index_select(
            current_encoder_out,
            dim=0,
            index=hyps_shape.row_ids(1).to(torch.int64),
        )  # (num_hyps, 1, 1, encoder_out_dim)

        logits = model.joiner(
            current_encoder_out, decoder_out, project_input=False
        )  # (num_hyps, 1, 1, vocab_size)
        logits = logits.squeeze(1).squeeze(1)  # (num_hyps, vocab_size)

        log_probs = logits.log_softmax(dim=-1)  # (num_hyps, vocab_size)
        log_probs.add_(ys_log_probs)

        vocab_size = log_probs.size(-1)

        row_splits = hyps_shape.row_splits(1) * vocab_size
        log_probs_shape = k2.ragged.create_ragged_shape2(
            row_splits=row_splits, cached_tot_size=log_probs.numel()
        )
        ragged_log_probs = k2.RaggedTensor(
            shape=log_probs_shape, value=log_probs.reshape(-1)
        )  # [batch][num_hyps*vocab_size]

        for i in range(batch_size):
            for h, hyp in enumerate(A[i]):
                pos_u = len(hyp.timestamp)
                idx_offset = h * vocab_size
                if (sorted_encoder_out_lens[i] - 1 - t) >= (sorted_ys_lens[i] - pos_u):
                    # emit blank token
                    new_hyp = Hypothesis(
                        log_prob=ragged_log_probs[i][idx_offset + blank_id],
                        ys=hyp.ys[:],
                        timestamp=hyp.timestamp[:],
                    )
                    B[i].add(new_hyp)
                if pos_u < sorted_ys_lens[i]:
                    # emit non-blank token
                    new_token = sorted_ys_list[i][pos_u]
                    new_hyp = Hypothesis(
                        log_prob=ragged_log_probs[i][idx_offset + new_token],
                        ys=hyp.ys + [new_token],
                        timestamp=hyp.timestamp + [t],
                    )
                    B[i].add(new_hyp)

            if len(B[i]) > beam_size:
                B[i] = B[i].topk(beam_size, length_norm=True)

    B = B + finalized_B
    sorted_hyps = [b.get_most_probable() for b in B]
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    hyps = [sorted_hyps[i] for i in unsorted_indices]
    ans = []
    for i, hyp in enumerate(hyps):
        assert hyp.ys[context_size:] == ys_list[i], (hyp.ys[context_size:], ys_list[i])
        ans.append(hyp.timestamp)

    return ans


def align_one_batch(
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    batch: dict,
) -> Tuple[List[List[str]], List[List[float]]]:
    """Get forced-alignments for one batch.

    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.

    Returns:
      token_list:
        A list of token list.
      token_time_list:
        A list of timestamps list for tokens.

      where len(token_list) == len(token_time_list),
    """
    device = next(model.parameters()).device
    feature = batch["inputs"]
    assert feature.ndim == 3

    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    encoder_out, encoder_out_lens = model.encoder(
        features=feature,
        feature_lengths=feature_lens,
    )

    texts = supervisions["text"]
    ys_list: List[List[int]] = sp.encode(texts, out_type=int)

    frame_indexes = batch_force_alignment(
        model, encoder_out, encoder_out_lens, ys_list, params.beam_size
    )

    token_list = []
    token_time_list = []
    for i in range(encoder_out.size(0)):
        tokens = sp.id_to_piece(ys_list[i])
        token_time = convert_timestamp(
            frame_indexes[i], params.subsampling_factor, params.frame_shift_ms
        )

        token_list.append(tokens)
        token_time_list.append(token_time)

    return token_list, token_time_list


def align_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    writer: SequentialJsonlWriter,
) -> None:
    """Get forced-alignments for the dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      writer:
        Writer to save the cuts with alignments.
    """
    log_interval = 20
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    for batch_idx, batch in enumerate(dl):
        token_list, token_time_list = align_one_batch(
            params=params, model=model, sp=sp, batch=batch
        )

        cut_list = batch["supervisions"]["cut"]
        for cut, token, token_time in zip(cut_list, token_list, token_time_list):
            assert len(cut.supervisions) == 1, f"{len(cut.supervisions)}"
            token_time += [cut.supervisions[0].duration]

            durations = [
                token_time[i] - token_time[i - 1] for i in range(1, len(token_time))
            ]

            assert len(token) == len(durations), (len(token), len(durations))

            cut.supervisions[0].durations = durations
            cut = cut.resample(sampling_rate=22050)

            del cut.recording.transforms

            writer.write(cut, flush=True)

        num_cuts += len(cut_list)
        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"
            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")


@torch.no_grad()
def main():
    parser = get_parser()
    AsrDataModule.add_arguments(parser)
    args = parser.parse_args()

    params = get_params()
    params.update(vars(args))

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> and <unk> are defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = torch.jit.load(params.checkpoint)
    model.to(device)
    model.eval()

    ljspeech = AsrDataModule(args)

    def normalize_text(c: Cut):
        normalized_text = c.supervisions[0].normalized_text
        c.supervisions[0].text = normalized_text
        return c

    cuts = load_manifest_lazy(params.manifests_in)
    cuts = cuts.map(normalize_text)
    cuts = cuts.resample(sampling_rate=16000)

    dl = ljspeech.dataloader(cuts)

    with CutSet.open_writer(params.manifests_out) as writer:
        align_dataset(dl=dl, params=params, model=model, sp=sp, writer=writer)

    logging.info(
        f"The cut manifest with framewise token alignments "
        f"and word alignments are saved to {params.manifests_out}"
    )
    logging.info("Done!")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
