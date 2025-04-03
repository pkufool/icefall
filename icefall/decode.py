# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                    Wei Kang)
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
import sys
from dataclasses import dataclass, field
from multiprocessing.pool import Pool
from typing import Dict, List, Optional, Tuple, Union

import torch

from icefall.context_graph import ContextGraph, ContextState
from icefall.lm_wrapper import LmScorer
from icefall.ngram_lm import NgramLm, NgramLmStateCost

if "k2" in sys.modules:
    from k2_decode import (
        get_lattice,
        Nbest,
        one_best_decoding,
        nbest_decoding,
        nbest_oracle,
        rescore_with_n_best_list,
        rescore_with_whole_lattice,
        nbest_rescore_with_LM,
        rescore_with_attention_decoder,
        rescore_with_attention_decoder_with_ngram,
        rescore_with_attention_decoder_no_ngram,
        rescore_with_rnn_lm,
    )
else:
    print("Warning: k2 is not installed. Some decoding methods might not work.")


def ctc_greedy_search(
    ctc_output: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    blank_id: int = 0,
) -> List[List[int]]:
    """CTC greedy search.

    Args:
         ctc_output: (batch, seq_len, vocab_size)
         encoder_out_lens: (batch,)
    Returns:
         List[List[int]]: greedy search result
    """
    batch = ctc_output.shape[0]
    index = ctc_output.argmax(dim=-1)  # (batch, seq_len)
    hyps = [
        torch.unique_consecutive(index[i, : encoder_out_lens[i]]) for i in range(batch)
    ]

    hyps = [h[h != blank_id].tolist() for h in hyps]

    return hyps


@dataclass
class Hypothesis:
    # The predicted tokens so far.
    # Newly predicted tokens are appended to `ys`.
    ys: List[int] = field(default_factory=list)

    # The log prob of ys that ends with blank token.
    # It contains only one entry.
    log_prob_blank: torch.Tensor = torch.zeros(1, dtype=torch.float32)

    # The log prob of ys that ends with non blank token.
    # It contains only one entry.
    log_prob_non_blank: torch.Tensor = torch.tensor(
        [float("-inf")], dtype=torch.float32
    )

    # timestamp[i] is the frame index after subsampling
    # on which ys[i] is decoded
    timestamp: List[int] = field(default_factory=list)

    # The lm score of ys
    # May contain external LM score (including LODR score) and contextual biasing score
    # It contains only one entry
    lm_score: torch.Tensor = torch.zeros(1, dtype=torch.float32)

    # the lm log_probs for next token given the history ys
    # The number of elements should be equal to vocabulary size.
    lm_log_probs: Optional[torch.Tensor] = None

    # the RNNLM states (h and c in LSTM)
    state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    # LODR (N-gram LM) state
    LODR_state: Optional[NgramLmStateCost] = None

    # N-gram LM state
    Ngram_state: Optional[NgramLmStateCost] = None

    # Context graph state
    context_state: Optional[ContextState] = None

    # This is the total score of current path, acoustic plus external LM score.
    @property
    def tot_score(self) -> torch.Tensor:
        return self.log_prob + self.lm_score

    # This is only the probability from model output (i.e External LM score not included).
    @property
    def log_prob(self) -> torch.Tensor:
        return torch.logaddexp(self.log_prob_non_blank, self.log_prob_blank)

    @property
    def key(self) -> tuple:
        """Return a tuple representation of self.ys"""
        return tuple(self.ys)

    def clone(self) -> "Hypothesis":
        return Hypothesis(
            ys=self.ys,
            log_prob_blank=self.log_prob_blank,
            log_prob_non_blank=self.log_prob_non_blank,
            timestamp=self.timestamp,
            lm_log_probs=self.lm_log_probs,
            lm_score=self.lm_score,
            state=self.state,
            LODR_state=self.LODR_state,
            Ngram_state=self.Ngram_state,
            context_state=self.context_state,
        )


class HypothesisList(object):
    def __init__(self, data: Optional[Dict[tuple, Hypothesis]] = None) -> None:
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
    def data(self) -> Dict[tuple, Hypothesis]:
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
            torch.logaddexp(
                old_hyp.log_prob_blank, hyp.log_prob_blank, out=old_hyp.log_prob_blank
            )
            torch.logaddexp(
                old_hyp.log_prob_non_blank,
                hyp.log_prob_non_blank,
                out=old_hyp.log_prob_non_blank,
            )
        else:
            self._data[key] = hyp

    def get_most_probable(self, length_norm: bool = False) -> Hypothesis:
        """Get the most probable hypothesis, i.e., the one with
        the largest `tot_score`.
        Args:
          length_norm:
            If True, the `tot_score` of a hypothesis is normalized by the
            number of tokens in it.
        Returns:
          Return the hypothesis that has the largest `tot_score`.
        """
        if length_norm:
            return max(self._data.values(), key=lambda hyp: hyp.tot_score / len(hyp.ys))
        else:
            return max(self._data.values(), key=lambda hyp: hyp.tot_score)

    def remove(self, hyp: Hypothesis) -> None:
        """Remove a given hypothesis.
        Caution:
          `self` is modified **in-place**.
        Args:
          hyp:
            The hypothesis to be removed from `self`.
            Note: It must be contained in `self`. Otherwise,
            an exception is raised.
        """
        key = hyp.key
        assert key in self, f"{key} does not exist"
        del self._data[key]

    def filter(self, threshold: torch.Tensor) -> "HypothesisList":
        """Remove all Hypotheses whose tot_score is less than threshold.
        Caution:
          `self` is not modified. Instead, a new HypothesisList is returned.
        Returns:
          Return a new HypothesisList containing all hypotheses from `self`
          with `tot_score` being greater than the given `threshold`.
        """
        ans = HypothesisList()
        for _, hyp in self._data.items():
            if hyp.tot_score > threshold:
                ans.add(hyp)  # shallow copy
        return ans

    def topk(self, k: int, length_norm: bool = False) -> "HypothesisList":
        """Return the top-k hypothesis.
        Args:
          length_norm:
            If True, the `tot_score` of a hypothesis is normalized by the
            number of tokens in it.
        """
        hyps = list(self._data.items())

        if length_norm:
            hyps = sorted(
                hyps, key=lambda h: h[1].tot_score / len(h[1].ys), reverse=True
            )[:k]
        else:
            hyps = sorted(hyps, key=lambda h: h[1].tot_score, reverse=True)[:k]

        ans = HypothesisList(dict(hyps))
        return ans

    def __contains__(self, key: tuple):
        return key in self._data

    def __getitem__(self, key: tuple):
        return self._data[key]

    def __iter__(self):
        return iter(self._data.values())

    def __len__(self) -> int:
        return len(self._data)

    def __str__(self) -> str:
        s = []
        for key in self:
            s.append(key)
        return ", ".join(str(s))


def _get_row_splits_row_ids(
    hyps: List[HypothesisList],
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the row_splits and row_ids of hyps.

    Suppose the hyps has 3 HypothesisList and each HypothesisList has
    2, 3 and 4 hypotheses respectively. Then the row_splits and row_ids
    are:
      row_splits = [0, 2, 5, 9]
      row_ids = [0, 0, 1, 1, 1, 2, 2, 2, 2]

    Args:
      hyps:
        len(hyps) == batch_size. It contains the current hypothesis for
        each utterance in the batch.
    Returns:
      Return row_splits and row_ids of hyps.
    """
    num_hyps = [len(h) for h in hyps]

    # torch.cumsum() is inclusive sum, so we put a 0 at the beginning
    # to get exclusive sum later.
    num_hyps.insert(0, 0)

    num_hyps = torch.tensor(num_hyps, device=device)
    row_splits = torch.cumsum(num_hyps, dim=0, dtype=torch.int32)

    row_ids = torch.zeros(num_hyps.sum(), dtype=torch.int32, device=device)
    for i in range(len(num_hyps) - 1):
        row_ids[row_splits[i] : row_splits[i + 1]] = i

    return row_splits, row_ids


def _get_max_indexes(scores: torch.Tensor, row_splits: torch.Tensor) -> torch.Tensor:
    """
    Get the local max indexes of scores with given row_splits.
    Args:
      scores:
        The scores to be processed, the shape is (N,)
      row_splits:
        The row_splits of scores which is used to split the scores into segments.
        The shape is (B + 1,). B is the number of segments. row_splits[-1] is
        equal to N.
    Returns:
      Return the local max indexes of each segment.
    """
    max_indexes = torch.zeros(
        len(row_splits) - 1, dtype=torch.int32, device=scores.device
    )
    for i in range(len(row_splits) - 1):
        start = row_splits[i]
        end = row_splits[i + 1]
        max_indexes[i] = torch.argmax(scores[start:end])
    return max_indexes


def _step_worker(
    log_probs: torch.Tensor,
    indexes: torch.Tensor,
    B: HypothesisList,
    beam: int = 4,
    blank_id: int = 0,
    nnlm_scale: float = 0,
    LODR_lm_scale: float = 0,
    context_graph: Optional[ContextGraph] = None,
) -> HypothesisList:
    """The worker to decode one step.
    Args:
      log_probs:
        topk log_probs of current step (i.e. the kept tokens of first pass pruning),
        the shape is (beam,)
      topk_indexes:
        The indexes of the topk_values above, the shape is (beam,)
      B:
        An instance of HypothesisList containing the kept hypothesis.
      beam:
        The number of hypothesis to be kept at each step.
      blank_id:
        The id of blank in the vocabulary.
      lm_scale:
        The scale of nn lm.
      LODR_lm_scale:
        The scale of the LODR_lm
      context_graph:
        A ContextGraph instance containing contextual phrases.
    Return:
      Returns the updated HypothesisList.
    """
    A = list(B)
    B = HypothesisList()
    for h in range(len(A)):
        hyp = A[h]
        for k in range(log_probs.size(0)):
            log_prob, index = log_probs[k], indexes[k]
            new_token = index.item()
            update_prefix = False
            new_hyp = hyp.clone()
            if new_token == blank_id:
                # Case 0: *a + ε => *a
                #         *aε + ε => *a
                # Prefix does not change, update log_prob of blank
                new_hyp.log_prob_non_blank = torch.tensor(
                    [float("-inf")], dtype=torch.float32
                )
                new_hyp.log_prob_blank = hyp.log_prob + log_prob
                B.add(new_hyp)
            elif len(hyp.ys) > 0 and hyp.ys[-1] == new_token:
                # Case 1: *a + a => *a
                # Prefix does not change, update log_prob of non_blank
                new_hyp.log_prob_non_blank = hyp.log_prob_non_blank + log_prob
                new_hyp.log_prob_blank = torch.tensor(
                    [float("-inf")], dtype=torch.float32
                )
                B.add(new_hyp)

                # Case 2: *aε + a => *aa
                # Prefix changes, update log_prob of blank
                new_hyp = hyp.clone()
                # Caution: DO NOT use append, as clone is shallow copy
                new_hyp.ys = hyp.ys + [new_token]
                new_hyp.log_prob_non_blank = hyp.log_prob_blank + log_prob
                new_hyp.log_prob_blank = torch.tensor(
                    [float("-inf")], dtype=torch.float32
                )
                update_prefix = True
            else:
                # Case 3: *a + b => *ab, *aε + b => *ab
                # Prefix changes, update log_prob of non_blank
                # Caution: DO NOT use append, as clone is shallow copy
                new_hyp.ys = hyp.ys + [new_token]
                new_hyp.log_prob_non_blank = hyp.log_prob + log_prob
                new_hyp.log_prob_blank = torch.tensor(
                    [float("-inf")], dtype=torch.float32
                )
                update_prefix = True

            if update_prefix:
                lm_score = hyp.lm_score
                if hyp.lm_log_probs is not None:
                    lm_score = lm_score + hyp.lm_log_probs[new_token] * nnlm_scale
                    new_hyp.lm_log_probs = None

                if context_graph is not None and hyp.context_state is not None:
                    (
                        context_score,
                        new_context_state,
                        matched_state,
                    ) = context_graph.forward_one_step(hyp.context_state, new_token)
                    lm_score = lm_score + context_score
                    new_hyp.context_state = new_context_state

                if hyp.LODR_state is not None:
                    state_cost = hyp.LODR_state.forward_one_step(new_token)
                    # calculate the score of the latest token
                    current_ngram_score = state_cost.lm_score - hyp.LODR_state.lm_score
                    assert current_ngram_score <= 0.0, (
                        state_cost.lm_score,
                        hyp.LODR_state.lm_score,
                    )
                    lm_score = lm_score + LODR_lm_scale * current_ngram_score
                    new_hyp.LODR_state = state_cost

                new_hyp.lm_score = lm_score
                B.add(new_hyp)
    B = B.topk(beam)
    return B


def _sequence_worker(
    topk_values: torch.Tensor,
    topk_indexes: torch.Tensor,
    B: HypothesisList,
    encoder_out_lens: torch.Tensor,
    beam: int = 4,
    blank_id: int = 0,
) -> HypothesisList:
    """The worker to decode one sequence.
    Args:
      topk_values:
        topk log_probs of model output (i.e. the kept tokens of first pass pruning),
        the shape is (T, beam)
      topk_indexes:
        The indexes of the topk_values above, the shape is (T, beam)
      B:
        An instance of HypothesisList containing the kept hypothesis.
      encoder_out_lens:
        The lengths (frames) of sequences after subsampling, the shape is (B,)
      beam:
        The number of hypothesis to be kept at each step.
      blank_id:
        The id of blank in the vocabulary.
    Return:
      Returns the updated HypothesisList.
    """
    B.add(Hypothesis())
    for j in range(encoder_out_lens):
        log_probs, indexes = topk_values[j], topk_indexes[j]
        B = _step_worker(log_probs, indexes, B, beam, blank_id)
    return B


def ctc_prefix_beam_search(
    ctc_output: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: int = 4,
    blank_id: int = 0,
    process_pool: Optional[Pool] = None,
    return_nbest: Optional[bool] = False,
) -> Union[List[List[int]], List[HypothesisList]]:
    """Implement prefix search decoding in "Connectionist Temporal Classification:
    Labelling Unsegmented Sequence Data with Recurrent Neural Networks".
    Args:
      ctc_output:
        The output of ctc head (log probability), the shape is (B, T, V)
      encoder_out_lens:
        The lengths (frames) of sequences after subsampling, the shape is (B,)
      beam:
        The number of hypothesis to be kept at each step.
      blank_id:
        The id of blank in the vocabulary.
      process_pool:
        The process pool for parallel decoding, if not provided, it will use all
        you cpu cores by default.
      return_nbest:
        If true, return a list of HypothesisList, return a list of list of decoded token ids otherwise.
    """
    batch_size, num_frames, vocab_size = ctc_output.shape

    # TODO: using a larger beam for first pass pruning
    topk_values, topk_indexes = ctc_output.topk(beam)  # (B, T, beam)
    topk_values = topk_values.cpu()
    topk_indexes = topk_indexes.cpu()

    B = [HypothesisList() for _ in range(batch_size)]

    pool = Pool() if process_pool is None else process_pool
    arguments = []
    for i in range(batch_size):
        arguments.append(
            (
                topk_values[i],
                topk_indexes[i],
                B[i],
                encoder_out_lens[i].item(),
                beam,
                blank_id,
            )
        )
    async_results = pool.starmap_async(_sequence_worker, arguments)
    B = list(async_results.get())
    if process_pool is None:
        pool.close()
        pool.join()
    if return_nbest:
        return B
    else:
        best_hyps = [b.get_most_probable() for b in B]
        return [hyp.ys for hyp in best_hyps]


def ctc_prefix_beam_search_shallow_fussion(
    ctc_output: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: int = 4,
    blank_id: int = 0,
    LODR_lm: Optional[NgramLm] = None,
    LODR_lm_scale: Optional[float] = 0,
    NNLM: Optional[LmScorer] = None,
    context_graph: Optional[ContextGraph] = None,
) -> List[List[int]]:
    """Implement prefix search decoding in "Connectionist Temporal Classification:
    Labelling Unsegmented Sequence Data with Recurrent Neural Networks" and add
    nervous language model shallow fussion, it also supports contextual
    biasing with a given grammar.
    Args:
      ctc_output:
        The output of ctc head (log probability), the shape is (B, T, V)
      encoder_out_lens:
        The lengths (frames) of sequences after subsampling, the shape is (B,)
      beam:
        The number of hypothesis to be kept at each step.
      blank_id:
        The id of blank in the vocabulary.
      LODR_lm:
        A low order n-gram LM, whose score will be subtracted during shallow fusion
      LODR_lm_scale:
        The scale of the LODR_lm
      LM:
        A neural net LM, e.g an RNNLM or transformer LM
      context_graph:
        A ContextGraph instance containing contextual phrases.
    Return:
      Returns a list of list of decoded token ids.
    """
    batch_size, num_frames, vocab_size = ctc_output.shape
    # TODO: using a larger beam for first pass pruning
    topk_values, topk_indexes = ctc_output.topk(beam)  # (B, T, beam)
    topk_values = topk_values.cpu()
    topk_indexes = topk_indexes.cpu()
    encoder_out_lens = encoder_out_lens.tolist()
    device = ctc_output.device

    nnlm_scale = 0
    init_scores = None
    init_states = None
    if NNLM is not None:
        nnlm_scale = NNLM.lm_scale
        sos_id = getattr(NNLM, "sos_id", 1)
        # get initial lm score and lm state by scoring the "sos" token
        sos_token = torch.tensor([[sos_id]]).to(torch.int64).to(device)
        lens = torch.tensor([1]).to(device)
        init_scores, init_states = NNLM.score_token(sos_token, lens)
        init_scores, init_states = init_scores.cpu(), (
            init_states[0].cpu(),
            init_states[1].cpu(),
        )

    B = [HypothesisList() for _ in range(batch_size)]
    for i in range(batch_size):
        B[i].add(
            Hypothesis(
                ys=[],
                log_prob_non_blank=torch.tensor([float("-inf")], dtype=torch.float32),
                log_prob_blank=torch.zeros(1, dtype=torch.float32),
                lm_score=torch.zeros(1, dtype=torch.float32),
                state=init_states,
                lm_log_probs=None if init_scores is None else init_scores.reshape(-1),
                LODR_state=None if LODR_lm is None else NgramLmStateCost(LODR_lm),
                context_state=None if context_graph is None else context_graph.root,
            )
        )
    for j in range(num_frames):
        for i in range(batch_size):
            if j < encoder_out_lens[i]:
                log_probs, indexes = topk_values[i][j], topk_indexes[i][j]
                B[i] = _step_worker(
                    log_probs=log_probs,
                    indexes=indexes,
                    B=B[i],
                    beam=beam,
                    blank_id=blank_id,
                    nnlm_scale=nnlm_scale,
                    LODR_lm_scale=LODR_lm_scale,
                    context_graph=context_graph,
                )
        if NNLM is None:
            continue
        # update lm_log_probs
        token_list = []  # a list of list
        hs = []
        cs = []
        indexes = []  # (batch_idx, key)
        for batch_idx, hyps in enumerate(B):
            for hyp in hyps:
                if hyp.lm_log_probs is None:  # those hyps that prefix changes
                    if NNLM.lm_type == "rnn":
                        token_list.append([hyp.ys[-1]])
                        # store the LSTM states
                        hs.append(hyp.state[0])
                        cs.append(hyp.state[1])
                    else:
                        # for transformer LM
                        token_list.append([sos_id] + hyp.ys[:])
                    indexes.append((batch_idx, hyp.key))
        if len(token_list) != 0:
            x_lens = torch.tensor([len(tokens) for tokens in token_list]).to(device)
            if NNLM.lm_type == "rnn":
                tokens_to_score = (
                    torch.tensor(token_list).to(torch.int64).to(device).reshape(-1, 1)
                )
                hs = torch.cat(hs, dim=1).to(device)
                cs = torch.cat(cs, dim=1).to(device)
                state = (hs, cs)
            else:
                # for transformer LM
                tokens_list = [torch.tensor(tokens) for tokens in token_list]
                tokens_to_score = (
                    torch.nn.utils.rnn.pad_sequence(
                        tokens_list, batch_first=True, padding_value=0.0
                    )
                    .to(device)
                    .to(torch.int64)
                )
                state = None

            scores, lm_states = NNLM.score_token(tokens_to_score, x_lens, state)
            scores, lm_states = scores.cpu(), (lm_states[0].cpu(), lm_states[1].cpu())
            assert scores.size(0) == len(indexes), (scores.size(0), len(indexes))
            for i in range(scores.size(0)):
                batch_idx, key = indexes[i]
                B[batch_idx][key].lm_log_probs = scores[i]
                if NNLM.lm_type == "rnn":
                    state = (
                        lm_states[0][:, i, :].unsqueeze(1),
                        lm_states[1][:, i, :].unsqueeze(1),
                    )
                    B[batch_idx][key].state = state

    # finalize context_state, if the matched contexts do not reach final state
    # we need to add the score on the corresponding backoff arc
    if context_graph is not None:
        for hyps in B:
            for hyp in hyps:
                context_score, new_context_state = context_graph.finalize(
                    hyp.context_state
                )
                hyp.lm_score += context_score
                hyp.context_state = new_context_state

    best_hyps = [b.get_most_probable() for b in B]
    return [hyp.ys for hyp in best_hyps]


def ctc_prefix_beam_search_attention_decoder_rescoring(
    ctc_output: torch.Tensor,
    attention_decoder: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: int = 8,
    blank_id: int = 0,
    attention_scale: Optional[float] = None,
    process_pool: Optional[Pool] = None,
):
    """Implement prefix search decoding in "Connectionist Temporal Classification:
    Labelling Unsegmented Sequence Data with Recurrent Neural Networks" and add
    attention decoder rescoring.
    Args:
      ctc_output:
        The output of ctc head (log probability), the shape is (B, T, V)
      attention_decoder:
        The attention decoder.
      encoder_out:
        The output of encoder, the shape is (B, T, D)
      encoder_out_lens:
        The lengths (frames) of sequences after subsampling, the shape is (B,)
      beam:
        The number of hypothesis to be kept at each step.
      blank_id:
        The id of blank in the vocabulary.
      attention_scale:
        The scale of attention decoder score, if not provided it will search in
        a default list (see the code below).
      process_pool:
        The process pool for parallel decoding, if not provided, it will use all
        you cpu cores by default.
    """
    # List[HypothesisList]
    nbest = ctc_prefix_beam_search(
        ctc_output=ctc_output,
        encoder_out_lens=encoder_out_lens,
        beam=beam,
        blank_id=blank_id,
        return_nbest=True,
    )

    device = ctc_output.device

    row_splits, row_ids = _get_row_splits_row_ids(nbest, device=device)
    hyp_to_utt_map = row_ids.to(torch.long)
    # the shape of encoder_out is (N, T, C), so we use axis=0 here
    expanded_encoder_out = encoder_out.index_select(0, hyp_to_utt_map)
    expanded_encoder_out_lens = encoder_out_lens.index_select(0, hyp_to_utt_map)

    nbest = [list(x) for x in nbest]
    token_ids = []
    scores = []
    for hyps in nbest:
        for hyp in hyps:
            token_ids.append(hyp.ys)
            scores.append(hyp.log_prob.reshape(1))
    scores = torch.cat(scores).to(device)

    nll = attention_decoder.nll(
        encoder_out=expanded_encoder_out,
        encoder_out_lens=expanded_encoder_out_lens,
        token_ids=token_ids,
    )
    assert nll.ndim == 2
    assert nll.shape[0] == len(token_ids)

    attention_scores = -nll.sum(dim=1)

    if attention_scale is None:
        attention_scale_list = [0.01, 0.05, 0.08]
        attention_scale_list += [0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 1.0]
        attention_scale_list += [1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.0]
        attention_scale_list += [2.1, 2.2, 2.3, 2.5, 3.0, 4.0, 5.0]
        attention_scale_list += [5.0, 6.0, 7.0, 8.0, 9.0]
    else:
        attention_scale_list = [attention_scale]

    ans = dict()

    for a_scale in attention_scale_list:
        tot_scores = scores + a_scale * attention_scores
        max_indexes = _get_max_indexes(tot_scores, row_splits)
        max_indexes = max_indexes.cpu()
        best_path = [nbest[i][max_indexes[i]].ys for i in range(len(max_indexes))]
        key = f"attention_scale_{a_scale}"
        ans[key] = best_path
    return ans
