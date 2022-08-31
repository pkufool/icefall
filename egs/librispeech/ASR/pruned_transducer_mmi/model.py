# Copyright    2022  Xiaomi Corp.        (author: Wei Kang)
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


import k2
import logging
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from encoder_interface import EncoderInterface
from scaling import ScaledLinear
from typing import Tuple

from icefall.utils import add_sos


def _roll_by_shifts(
    src: torch.Tensor, shifts: torch.LongTensor
) -> torch.Tensor:
    """Roll tensor with different shifts for each row.
    Note:
      We assume the src is a 3 dimensions tensor and roll the last dimension.
    Example:
      >>> src = torch.arange(15).reshape((1,3,5))
      >>> src
      tensor([[[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14]]])
      >>> shift = torch.tensor([[1, 2, 3]])
      >>> shift
      tensor([[1, 2, 3]])
      >>> _roll_by_shifts(src, shift)
      tensor([[[ 4,  0,  1,  2,  3],
               [ 8,  9,  5,  6,  7],
               [12, 13, 14, 10, 11]]])
    """
    assert src.dim() == 3
    (B, T, S) = src.shape
    assert shifts.shape == (B, T)

    index = (
        torch.arange(S, device=src.device)
        .view((1, S))
        .repeat((T, 1))
        .repeat((B, 1, 1))
    )
    index = (index - shifts.reshape(B, T, 1)) % S
    return torch.gather(src, 2, index)


class Transducer(nn.Module):
    def __init__(
        self,
        encoder: EncoderInterface,
        hybrid_decoder: nn.Module,
        hybrid_joiner: nn.Module,
        predictor_decoder: nn.Module,
        predictor_joiner: nn.Module,
        external_lm: nn.Module,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
    ):
        """
        Args:
          encoder:
            The shared transcription network. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dm) and
            `logit_lens` of shape (N,).
          hybrid_decoder:
            The prediction network of hybrid head. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
          hybrid_joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and
            (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output
            contains unnormalized probs, i.e., not processed by log-softmax.
          predictor_decoder:
            The prediction network of predictor head. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
          predictor_joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and
            (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output
            contains unnormalized probs, i.e., not processed by log-softmax.
          encoder_dim:
            The output dimension of encoder.
          decoder_dim:
            The output dimension of decoder.
          joiner_dim:
            The output dimension of joiner.
          vocab_size:
            The vocabulary size.
        """
        super().__init__()
        assert isinstance(encoder, EncoderInterface), type(encoder)
        assert hasattr(hybrid_decoder, "blank_id")
        assert hasattr(predictor_decoder, "blank_id")

        self.encoder = encoder
        self.hybrid_decoder = hybrid_decoder
        self.hybrid_joiner = hybrid_joiner
        self.predictor_decoder = predictor_decoder
        self.predictor_joiner = predictor_joiner

        self.simple_am_proj = ScaledLinear(
            encoder_dim, vocab_size, initial_speed=0.5
        )
        self.simple_predictor_lm_proj = ScaledLinear(decoder_dim, vocab_size)

    def importance_sampling(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        path_length: int,
        num_paths: int,
        normalized: bool = False,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Args:
          encoder_out:
            The output of the encoder whose shape is (batch_size, T, encoder_dim)
          encoder_out_lens:
            A tensor of shape (batch_size,) containing the number of frames
            before padding.
          path_length:
            How many symbols we will sample for each path.
          num_paths:
            How many paths we will sample for each sequence.
          normalized:
            True to normalize the `path_scores`, otherwise not.
        Returns:
          Five tensors will be returned.
          - sampled_paths:
            A tensor of shape (batch_size, num_paths, path_length), containing the
            sampled symbol ids.
          - sampling_probs:
            A tensor of shape (batch_size, num_paths, path_length), containing the
            sampling probabilities of the sampled symbols.
          - path_scores:
            A tensor of shape (batch_size, num_paths, path_length), containing the
            scores of the sampled paths, the scores include joiner output and
            external_lm output.
          - left_symbols:
            A tensor of shape (batch_size, num_paths, path_length, context_size),
            containing the left symbols of the sampled symbols.
          - frame_ids:
            A tensor of shape (batch_size, num_paths, path_length), containing the
            frame ids at which we sampled the symbols.
        """
        batch_size, T, encoder_dim = encoder_out.shape
        device = encoder_out.device

        blank_id = self.predictor_decoder.blank_id
        assert blank_id == self.hybrid_decoder.blank_id

        context_size = self.predictor_decoder.context_size
        assert context_size == self.hybrid_decoder.context_size

        decoder_dim = self.predictor_decoder.decoder_dim
        assert decoder_dim == self.hybrid_decoder.decoder_dim

        # (TODO:Wei Kang) Change to sampling different path from different
        # frame and use smaller path_length.
        # we sample paths from frame 0
        t_index = torch.zeros(
            (batch_size, num_paths), dtype=torch.int64, device=device
        )

        # The max frame index for each path
        t_index_max = encoder_out_lens.view(batch_size, 1).expand(
            batch_size, num_paths
        )

        # The left_symbols for each path
        left_symbols = torch.tensor(
            [blank_id], dtype=torch.int64, device=device
        ).expand(batch_size * num_paths, context_size)

        sampled_paths_list = []
        sampling_probs_list = []
        path_scores_list = []
        frame_ids_list = []
        left_symbols_list = []

        for i in range(path_length):
            # (B, num_paths, encoder_dim)
            current_encoder_out = torch.gather(
                encoder_out,
                dim=1,
                index=t_index.unsqueeze(2).expand(
                    batch_size, num_paths, encoder_dim
                ),
            )

            # (B, num_paths, decoder_dim)
            predictor_decoder_output = self.predictor_decoder(
                left_symbols, need_pad=False
            ).view(batch_size, num_paths, decoder_dim)
            # (B, num_paths, V)
            predictor_joiner_output = self.predictor_joiner(
                current_encoder_out, predictor_decoder_output
            )

            # (B, num_paths, decoder_dim)
            hybrid_decoder_output = self.hybrid_decoder(
                left_symbols, need_pad=False
            ).view(batch_size, num_paths, decoder_dim)
            # (B, num_paths, V)
            hybrid_joiner_output = self.hybrid_joiner(
                current_encoder_out, hybrid_decoder_output
            )

            probs = torch.softmax(predictor_joiner_output, -1)
            # sampler: https://pytorch.org/docs/stable/distributions.html#categorical
            sampler = Categorical(probs=probs)

            # sample one symbol for each path
            # index : (batch_size, num_paths)
            index = sampler.sample()
            sampled_paths_list.append(index)

            frame_ids_list.append(t_index)
            # update (t, s) for each path
            # index == 0 means the sampled symbol is blank
            t_mask = index == 0
            # t_index = torch.where(t_mask, t_index + 1, t_index)
            # we currently use modified like path, i.e. there is only one symbol
            # could be emitted at each frame
            t_index = t_index + 1

            final_mask = t_index >= t_index_max
            reach_final = torch.any(final_mask)
            # if reaching final frame, start from randomly selected frame id.
            if reach_final:
                min_t_index_max = torch.min(t_index_max) - 1
                min_t_index_max = 1 if min_t_index_max < 1 else min_t_index_max
                new_t_index = torch.randint(0, min_t_index_max, (1,)).item()
                t_index.masked_fill_(final_mask, new_t_index)

            left_symbols = left_symbols.view(
                batch_size, num_paths, context_size
            )
            left_symbols_list.append(left_symbols)
            current_symbols = torch.cat(
                [
                    left_symbols,
                    index.unsqueeze(2),
                ],
                dim=2,
            )
            # if the sampled symbol is blank, we only need to roll the history
            # symbols, if the sampled symbol is not blank, append the newly
            # sampled symbol.
            left_symbols = _roll_by_shifts(
                current_symbols, t_mask.to(torch.int64)
            )
            left_symbols = left_symbols[:, :, 1:]
            # when reaching final frames, we need to reset the left_symbols to
            # null.
            if reach_final:
                left_symbols.masked_fill_(final_mask.unsqueeze(2), blank_id)

            left_symbols = left_symbols.view(
                batch_size * num_paths, context_size
            )

            # gather sampling probabilities for corresponding indexs
            # sampling_prob : (batch_size, num_paths, 1)
            sampling_probs = torch.gather(
                probs, dim=2, index=index.unsqueeze(2)
            )

            sampling_probs_list.append(sampling_probs.squeeze(2))

            # (B, num_paths, 1)
            if normalized:
                hybrid_joiner_output = torch.nn.functional.log_softmax(hybrid_joiner_output, dim=2)
            hybrid_scores = torch.gather(
                hybrid_joiner_output, dim=2, index=index.unsqueeze(2)
            )
            path_scores_list.append(hybrid_scores)

        # sampled_paths : (batch_size, num_paths, path_lengths)
        sampled_paths = torch.stack(sampled_paths_list, dim=2).int()
        # sampling_probs : (batch_size, num_paths, path_lengths)
        sampling_probs = torch.stack(sampling_probs_list, dim=2)
        # path_scores : (batch_size, num_paths, path_lengths)
        path_scores = torch.stack(path_scores_list, dim=2)
        # frame_ids : (batch_size , num_paths, path_lengths)
        frame_ids = torch.stack(frame_ids_list, dim=2).int()
        # left_symbols : (batch_size, num_paths, path_lengths, context_size)
        left_symbols = torch.stack(left_symbols_list, dim=2).int()
        return (
            sampled_paths,
            frame_ids,
            sampling_probs,
            path_scores,
            left_symbols,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        path_length: int = 25,
        num_paths_per_frame: int = 10,
        normalized: bool = False,
        enable_den_loss: bool = False,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        warmup: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          path_length:
            How many symbols (including blank symbol) will be sampled for a
            linear path.
          num_paths_per_frame:
            How many linear paths will be sampled when generating the
            denominator lattice.
          normalized:
            Whether to normalize num loss and den loss.
          enable_den_loss:
            True to sample denominator lattice and return den loss, otherwise not.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
          warmup:
            A value warmup >= 0 that determines which modules are active, values
            warmup > 1 "are fully warmed up" and all modules will be active.
        Returns:
          Return the transducer loss.

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0

        # encoder_out : [B, T, encoder_dim]
        encoder_out, x_lens = self.encoder(x, x_lens, warmup=warmup)
        assert torch.all(x_lens > 0)

        max_len = torch.max(x_lens).item()

        # TODO(Wei Kang): Remove this line when finishing debuging
        # To make sure we won't sample an empty denominator lattice.
        path_length = max_len if max_len > path_length else path_length

        # Now for the decoder, i.e., the prediction network
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        blank_id = self.predictor_decoder.blank_id
        vocab_size = self.predictor_decoder.vocab_size
        context_size = self.predictor_decoder.context_size
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        # decoder_out: [B, S + 1, decoder_dim]
        predictor_decoder_out = self.predictor_decoder(sos_y_padded)
        hybrid_decoder_out = self.hybrid_decoder(sos_y_padded)

        # Note: y does not start with SOS
        # y_padded : [B, S]
        y_padded = y.pad(mode="constant", padding_value=0)

        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros(
            (x.size(0), 4), dtype=torch.int64, device=x.device
        )
        boundary[:, 2] = y_lens
        boundary[:, 3] = x_lens

        # pruned rnnt loss for predictor head
        predictor_lm = self.simple_predictor_lm_proj(predictor_decoder_out)
        am = self.simple_am_proj(encoder_out)

        with torch.cuda.amp.autocast(enabled=False):
            predictor_simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=predictor_lm.float(),
                am=am.float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                modified=True,
                reduction="sum",
                return_grad=True,
            )

        # ranges : [B, T, prune_range]
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.predictor_joiner.encoder_proj(encoder_out),
            lm=self.predictor_joiner.decoder_proj(predictor_decoder_out),
            ranges=ranges,
        )

        # predictor_logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        predictor_logits = self.predictor_joiner(
            am_pruned, lm_pruned, project_input=False
        )

        with torch.cuda.amp.autocast(enabled=False):
            predictor_pruned_loss = k2.rnnt_loss_pruned(
                logits=predictor_logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                modified=True,
                reduction="sum",
            )

        # pruned rnnt loss for hybrid head

        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.hybrid_joiner.encoder_proj(encoder_out),
            lm=self.hybrid_joiner.decoder_proj(hybrid_decoder_out),
            ranges=ranges,
        )

        # predictor_logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        hybrid_logits = self.hybrid_joiner(
            am_pruned, lm_pruned, project_input=False
        )

        with torch.cuda.amp.autocast(enabled=False):
            hybrid_pruned_loss = k2.rnnt_loss_pruned(
                logits=hybrid_logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                normalized=normalized,
                modified=True,
                reduction="sum",
            )

        den_loss = None
        posterior_loss = None
        if enable_den_loss:
            # Sampling denominator lattice
            (
                sampled_paths,
                frame_ids,
                sampling_probs,
                path_scores,
                left_symbols,
            ) = self.importance_sampling(
                encoder_out=encoder_out,
                encoder_out_lens=x_lens,
                path_length=path_length,
                num_paths=num_paths_per_frame,
                normalized=normalized,
            )

            den_lattice, arc_map = k2.generate_denominator_lattice(
                sampled_paths=sampled_paths,
                frame_ids=frame_ids,
                left_symbols=left_symbols,
                sampling_probs=sampling_probs.detach(),
                boundary=x_lens,
                path_scores=path_scores,
                vocab_size=vocab_size,
                context_size=context_size,
                return_arc_map=True,
            )

            # Only the final arc could be -1 in the arc_map
            den_lattice.predictor_logprobs = -torch.log(k2.index_select(sampling_probs.flatten(), arc_map, default_value=1.0))
            # predictor_logprobs will propagate properly
            den_lattice = k2.connect(k2.top_sort(den_lattice))

            posterior = den_lattice.get_arc_post(log_semiring=True, use_double_scores=True).detach()

            posterior_lattice = k2.Fsa(den_lattice.arcs)
            posterior_lattice.scores = (den_lattice.predictor_logprobs * posterior).float()
            posterior_scores = posterior_lattice.get_tot_scores(log_semiring=True, use_double_scores=True)

            posterior_loss = -torch.sum(posterior_scores)

            den_scores = den_lattice.get_tot_scores(
                log_semiring=True, use_double_scores=True
            )
            den_loss = -torch.sum(den_scores)

        return (
            hybrid_pruned_loss,
            den_loss,
            predictor_simple_loss,
            predictor_pruned_loss,
            posterior_loss
        )
