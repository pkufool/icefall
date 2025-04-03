# Copyright      2021-2024  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                        Mingshuang Luo,
#                                                        Zengwei Yao,
#                                                        Wei Kang)
#
# See ../../LICENSE for clarification regarding multiple authors
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


from typing import List, Optional, Tuple, Union

import k2
import sentencepiece as spm
import torch
import torch.nn as nn

from .symbol_table import SymbolTable


def get_texts(
    best_paths: k2.Fsa, return_ragged: bool = False
) -> Union[List[List[int]], k2.RaggedTensor]:
    """Extract the texts (as word IDs) from the best-path FSAs.
    Args:
      best_paths:
        A k2.Fsa with best_paths.arcs.num_axes() == 3, i.e.
        containing multiple FSAs, which is expected to be the result
        of k2.shortest_path (otherwise the returned values won't
        be meaningful).
      return_ragged:
        True to return a ragged tensor with two axes [utt][word_id].
        False to return a list-of-list word IDs.
    Returns:
      Returns a list of lists of int, containing the label sequences we
      decoded.
    """
    if isinstance(best_paths.aux_labels, k2.RaggedTensor):
        # remove 0's and -1's.
        aux_labels = best_paths.aux_labels.remove_values_leq(0)
        # TODO: change arcs.shape() to arcs.shape
        aux_shape = best_paths.arcs.shape().compose(aux_labels.shape)

        # remove the states and arcs axes.
        aux_shape = aux_shape.remove_axis(1)
        aux_shape = aux_shape.remove_axis(1)
        aux_labels = k2.RaggedTensor(aux_shape, aux_labels.values)
    else:
        # remove axis corresponding to states.
        aux_shape = best_paths.arcs.shape().remove_axis(1)
        aux_labels = k2.RaggedTensor(aux_shape, best_paths.aux_labels)
        # remove 0's and -1's.
        aux_labels = aux_labels.remove_values_leq(0)

    assert aux_labels.num_axes == 2
    if return_ragged:
        return aux_labels
    else:
        return aux_labels.tolist()


def get_texts_with_timestamp(
    best_paths: k2.Fsa, return_ragged: bool = False
) -> DecodingResults:
    """Extract the texts (as word IDs) and timestamps (as frame indexes)
    from the best-path FSAs.
    Args:
      best_paths:
        A k2.Fsa with best_paths.arcs.num_axes() == 3, i.e.
        containing multiple FSAs, which is expected to be the result
        of k2.shortest_path (otherwise the returned values won't
        be meaningful).
      return_ragged:
        True to return a ragged tensor with two axes [utt][word_id].
        False to return a list-of-list word IDs.
    Returns:
      Returns a list of lists of int, containing the label sequences we
      decoded.
    """
    if isinstance(best_paths.aux_labels, k2.RaggedTensor):
        all_aux_shape = (
            best_paths.arcs.shape().remove_axis(1).compose(best_paths.aux_labels.shape)
        )
        all_aux_labels = k2.RaggedTensor(all_aux_shape, best_paths.aux_labels.values)
        # remove 0's and -1's.
        aux_labels = best_paths.aux_labels.remove_values_leq(0)
        # TODO: change arcs.shape() to arcs.shape
        aux_shape = best_paths.arcs.shape().compose(aux_labels.shape)
        # remove the states and arcs axes.
        aux_shape = aux_shape.remove_axis(1)
        aux_shape = aux_shape.remove_axis(1)
        aux_labels = k2.RaggedTensor(aux_shape, aux_labels.values)
    else:
        # remove axis corresponding to states.
        aux_shape = best_paths.arcs.shape().remove_axis(1)
        all_aux_labels = k2.RaggedTensor(aux_shape, best_paths.aux_labels)
        # remove 0's and -1's.
        aux_labels = all_aux_labels.remove_values_leq(0)

    assert aux_labels.num_axes == 2

    timestamps = []
    if isinstance(best_paths.aux_labels, k2.RaggedTensor):
        for p in range(all_aux_labels.dim0):
            time = []
            for i, arc in enumerate(all_aux_labels[p].tolist()):
                if len(arc) == 1 and arc[0] > 0:
                    time.append(i)
            timestamps.append(time)
    else:
        for labels in all_aux_labels.tolist():
            time = [i for i, v in enumerate(labels) if v > 0]
            timestamps.append(time)

    return DecodingResults(
        timestamps=timestamps,
        hyps=aux_labels if return_ragged else aux_labels.tolist(),
    )


def get_alignments(best_paths: k2.Fsa, kind: str) -> List[List[int]]:
    """Extract labels or aux_labels from the best-path FSAs.

    Args:
      best_paths:
        A k2.Fsa with best_paths.arcs.num_axes() == 3, i.e.
        containing multiple FSAs, which is expected to be the result
        of k2.shortest_path (otherwise the returned values won't
        be meaningful).
      kind:
        Possible values are: "labels" and "aux_labels". Caution: When it is
        "labels", the resulting alignments contain repeats.
    Returns:
      Returns a list of lists of int, containing the token sequences we
      decoded. For `ans[i]`, its length equals to the number of frames
      after subsampling of the i-th utterance in the batch.

    Example:
      When `kind` is `labels`, one possible alignment example is (with
      repeats)::

        c c c blk a a blk blk t t t blk blk

     If `kind` is `aux_labels`, the above example changes to::

        c blk blk blk a blk blk blk t blk blk blk blk

    """
    assert kind in ("labels", "aux_labels")
    # arc.shape() has axes [fsa][state][arc], we remove "state"-axis here
    token_shape = best_paths.arcs.shape().remove_axis(1)
    # token_shape has axes [fsa][arc]
    tokens = k2.RaggedTensor(token_shape, getattr(best_paths, kind).contiguous())
    tokens = tokens.remove_values_eq(-1)
    return tokens.tolist()


def concat(ragged: k2.RaggedTensor, value: int, direction: str) -> k2.RaggedTensor:
    """Prepend a value to the beginning of each sublist or append a value.
    to the end of each sublist.

    Args:
      ragged:
        A ragged tensor with two axes.
      value:
        The value to prepend or append.
      direction:
        It can be either "left" or "right". If it is "left", we
        prepend the value to the beginning of each sublist;
        if it is "right", we append the value to the end of each
        sublist.

    Returns:
      Return a new ragged tensor, whose sublists either start with
      or end with the given value.

    >>> a = k2.RaggedTensor([[1, 3], [5]])
    >>> a
    [ [ 1 3 ] [ 5 ] ]
    >>> concat(a, value=0, direction="left")
    [ [ 0 1 3 ] [ 0 5 ] ]
    >>> concat(a, value=0, direction="right")
    [ [ 1 3 0 ] [ 5 0 ] ]

    """
    dtype = ragged.dtype
    device = ragged.device

    assert ragged.num_axes == 2, f"num_axes: {ragged.num_axes}"
    pad_values = torch.full(
        size=(ragged.tot_size(0), 1),
        fill_value=value,
        device=device,
        dtype=dtype,
    )
    pad = k2.RaggedTensor(pad_values)

    if direction == "left":
        ans = k2.ragged.cat([pad, ragged], axis=1)
    elif direction == "right":
        ans = k2.ragged.cat([ragged, pad], axis=1)
    else:
        raise ValueError(
            f'Unsupported direction: {direction}. " \
            "Expect either "left" or "right"'
        )
    return ans


def add_sos(ragged: k2.RaggedTensor, sos_id: int) -> k2.RaggedTensor:
    """Add SOS to each sublist.

    Args:
      ragged:
        A ragged tensor with two axes.
      sos_id:
        The ID of the SOS symbol.

    Returns:
      Return a new ragged tensor, where each sublist starts with SOS.

    >>> a = k2.RaggedTensor([[1, 3], [5]])
    >>> a
    [ [ 1 3 ] [ 5 ] ]
    >>> add_sos(a, sos_id=0)
    [ [ 0 1 3 ] [ 0 5 ] ]

    """
    return concat(ragged, sos_id, direction="left")


def add_eos(ragged: k2.RaggedTensor, eos_id: int) -> k2.RaggedTensor:
    """Add EOS to each sublist.

    Args:
      ragged:
        A ragged tensor with two axes.
      eos_id:
        The ID of the EOS symbol.

    Returns:
      Return a new ragged tensor, where each sublist ends with EOS.

    >>> a = k2.RaggedTensor([[1, 3], [5]])
    >>> a
    [ [ 1 3 ] [ 5 ] ]
    >>> add_eos(a, eos_id=0)
    [ [ 1 3 0 ] [ 5 0 ] ]

    """
    return concat(ragged, eos_id, direction="right")


def parse_bpe_timestamps_and_texts(
    best_paths: k2.Fsa, sp: spm.SentencePieceProcessor
) -> Tuple[List[Tuple[int, int]], List[List[str]]]:
    """Parse timestamps (frame indexes) and texts.

    Args:
      best_paths:
        A k2.Fsa with best_paths.arcs.num_axes() == 3, i.e.
        containing multiple FSAs, which is expected to be the result
        of k2.shortest_path (otherwise the returned values won't
        be meaningful). Its attributes `labels` and `aux_labels`
        are both BPE tokens.
      sp:
        The BPE model.

    Returns:
      utt_index_pairs:
        A list of pair list. utt_index_pairs[i] is a list of
        (start-frame-index, end-frame-index) pairs for each word in
        utterance-i.
      utt_words:
        A list of str list. utt_words[i] is a word list of utterence-i.
    """
    shape = best_paths.arcs.shape().remove_axis(1)

    # labels: [utt][arcs]
    labels = k2.RaggedTensor(shape, best_paths.labels.contiguous())
    # remove -1's.
    labels = labels.remove_values_eq(-1)
    labels = labels.tolist()

    # aux_labels: [utt][arcs]
    aux_labels = k2.RaggedTensor(shape, best_paths.aux_labels.contiguous())

    # remove -1's.
    all_aux_labels = aux_labels.remove_values_eq(-1)
    # len(all_aux_labels[i]) is equal to the number of frames
    all_aux_labels = all_aux_labels.tolist()

    # remove 0's and -1's.
    out_aux_labels = aux_labels.remove_values_leq(0)
    # len(out_aux_labels[i]) is equal to the number of output BPE tokens
    out_aux_labels = out_aux_labels.tolist()

    utt_index_pairs = []
    utt_words = []
    for i in range(len(labels)):
        tokens = sp.id_to_piece(labels[i])
        words = sp.decode(out_aux_labels[i]).split()

        # Indicates whether it is the first token, i.e., not-repeat and not-blank.
        is_first_token = [a != 0 for a in all_aux_labels[i]]
        index_pairs = parse_bpe_start_end_pairs(tokens, is_first_token)
        assert len(index_pairs) == len(words), (len(index_pairs), len(words), tokens)
        utt_index_pairs.append(index_pairs)
        utt_words.append(words)

    return utt_index_pairs, utt_words


def parse_timestamps_and_texts(
    best_paths: k2.Fsa, word_table: SymbolTable
) -> Tuple[List[Tuple[int, int]], List[List[str]]]:
    """Parse timestamps (frame indexes) and texts.

    Args:
      best_paths:
        A k2.Fsa with best_paths.arcs.num_axes() == 3, i.e.
        containing multiple FSAs, which is expected to be the result
        of k2.shortest_path (otherwise the returned values won't
        be meaningful). Attribute `labels` is the prediction unit,
        e.g., phone or BPE tokens. Attribute `aux_labels` is the word index.
      word_table:
        The word symbol table.

    Returns:
      utt_index_pairs:
        A list of pair list. utt_index_pairs[i] is a list of
        (start-frame-index, end-frame-index) pairs for each word in
        utterance-i.
      utt_words:
        A list of str list. utt_words[i] is a word list of utterence-i.
    """
    # [utt][words]
    word_ids = get_texts(best_paths)

    shape = best_paths.arcs.shape().remove_axis(1)

    # labels: [utt][arcs]
    labels = k2.RaggedTensor(shape, best_paths.labels.contiguous())
    # remove -1's.
    labels = labels.remove_values_eq(-1)
    labels = labels.tolist()

    # aux_labels: [utt][arcs]
    aux_shape = shape.compose(best_paths.aux_labels.shape)
    aux_labels = k2.RaggedTensor(aux_shape, best_paths.aux_labels.values.contiguous())
    aux_labels = aux_labels.tolist()

    utt_index_pairs = []
    utt_words = []
    for i, (label, aux_label) in enumerate(zip(labels, aux_labels)):
        num_arcs = len(label)
        # The last arc of aux_label is the arc entering the final state
        assert num_arcs == len(aux_label) - 1, (num_arcs, len(aux_label))

        index_pairs = []
        start = -1
        end = -1
        for arc in range(num_arcs):
            # len(aux_label[arc]) is 0 or 1
            if label[arc] != 0 and len(aux_label[arc]) != 0:
                if start != -1 and end != -1:
                    index_pairs.append((start, end))
                start = arc
            if label[arc] != 0:
                end = arc
        if start != -1 and end != -1:
            index_pairs.append((start, end))

        words = [word_table[w] for w in word_ids[i]]
        assert len(index_pairs) == len(words), (len(index_pairs), len(words))

        utt_index_pairs.append(index_pairs)
        utt_words.append(words)

    return utt_index_pairs, utt_words


def parse_fsa_timestamps_and_texts(
    best_paths: k2.Fsa,
    sp: Optional[spm.SentencePieceProcessor] = None,
    word_table: Optional[SymbolTable] = None,
    subsampling_factor: int = 4,
    frame_shift_ms: float = 10,
) -> Tuple[List[Tuple[float, float]], List[List[str]]]:
    """Parse timestamps (in seconds) and texts for given decoded fsa paths.
    Currently it supports two cases:
    (1) ctc-decoding, the attributes `labels` and `aux_labels`
        are both BPE tokens. In this case, sp should be provided.
    (2) HLG-based 1best, the attribtute `labels` is the prediction unit,
        e.g., phone or BPE tokens; attribute `aux_labels` is the word index.
        In this case, word_table should be provided.

    Args:
      best_paths:
        A k2.Fsa with best_paths.arcs.num_axes() == 3, i.e.
        containing multiple FSAs, which is expected to be the result
        of k2.shortest_path (otherwise the returned values won't
        be meaningful).
      sp:
        The BPE model.
      word_table:
        The word symbol table.
      subsampling_factor:
        The subsampling factor of the model.
      frame_shift_ms:
        Frame shift in milliseconds between two contiguous frames.

    Returns:
      utt_time_pairs:
        A list of pair list. utt_time_pairs[i] is a list of
        (start-time, end-time) pairs for each word in
        utterance-i.
      utt_words:
        A list of str list. utt_words[i] is a word list of utterence-i.
    """
    if sp is not None:
        assert word_table is None, "word_table is not needed if sp is provided."
        utt_index_pairs, utt_words = parse_bpe_timestamps_and_texts(
            best_paths=best_paths, sp=sp
        )
    elif word_table is not None:
        assert sp is None, "sp is not needed if word_table is provided."
        utt_index_pairs, utt_words = parse_timestamps_and_texts(
            best_paths=best_paths, word_table=word_table
        )
    else:
        raise ValueError("Either sp or word_table should be provided.")

    utt_time_pairs = []
    for utt in utt_index_pairs:
        start = convert_timestamp(
            frames=[i[0] for i in utt],
            subsampling_factor=subsampling_factor,
            frame_shift_ms=frame_shift_ms,
        )
        end = convert_timestamp(
            # The duration in frames is (end_frame_index - start_frame_index + 1)
            frames=[i[1] + 1 for i in utt],
            subsampling_factor=subsampling_factor,
            frame_shift_ms=frame_shift_ms,
        )
        utt_time_pairs.append(list(zip(start, end)))

    return utt_time_pairs, utt_words
