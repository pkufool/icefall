# Copyright    2022  Xiaomi Corp.        (authors: Daniel Povey)
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


import collections
from itertools import repeat
from typing import Optional, Tuple
import logging

import random
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Embedding as ScaledEmbedding


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)


class ActivationBalancerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        channel_dim: int,
        min_positive: float,  # e.g. 0.05
        max_positive: float,  # e.g. 0.95
        max_factor: float,  # e.g. 0.01
        min_abs: float,  # e.g. 0.2
        max_abs: float,  # e.g. 100.0
    ) -> Tensor:
        if x.requires_grad:
            if channel_dim < 0:
                channel_dim += x.ndim
            sum_dims = [d for d in range(x.ndim) if d != channel_dim]
            xgt0 = x > 0
            proportion_positive = torch.mean(
                xgt0.to(x.dtype), dim=sum_dims, keepdim=True
            )
            factor1 = (
                (min_positive - proportion_positive).relu()
                * (max_factor / min_positive)
                if min_positive != 0.0
                else 0.0
            )
            factor2 = (
                (proportion_positive - max_positive).relu()
                * (max_factor / (max_positive - 1.0))
                if max_positive != 1.0
                else 0.0
            )
            factor = factor1 + factor2
            if isinstance(factor, float):
                factor = torch.zeros_like(proportion_positive)

            mean_abs = torch.mean(x.abs(), dim=sum_dims, keepdim=True)
            below_threshold = mean_abs < min_abs
            above_threshold = mean_abs > max_abs

            ctx.save_for_backward(
                factor, xgt0, below_threshold, above_threshold
            )
            ctx.max_factor = max_factor
            ctx.sum_dims = sum_dims
        return x

    @staticmethod
    def backward(
        ctx, x_grad: Tensor
    ) -> Tuple[Tensor, None, None, None, None, None, None]:
        factor, xgt0, below_threshold, above_threshold = ctx.saved_tensors
        dtype = x_grad.dtype
        scale_factor = (
            (below_threshold.to(dtype) - above_threshold.to(dtype))
            * (xgt0.to(dtype) - 0.5)
            * (ctx.max_factor * 2.0)
        )

        neg_delta_grad = x_grad.abs() * (factor + scale_factor)
        return x_grad - neg_delta_grad, None, None, None, None, None, None


class BasicNorm(torch.nn.Module):
    """
    This is intended to be a simpler, and hopefully cheaper, replacement for
    LayerNorm.  The observation this is based on, is that Transformer-type
    networks, especially with pre-norm, sometimes seem to set one of the
    feature dimensions to a large constant value (e.g. 50), which "defeats"
    the LayerNorm because the output magnitude is then not strongly dependent
    on the other (useful) features.  Presumably the weight and bias of the
    LayerNorm are required to allow it to do this.

    So the idea is to introduce this large constant value as an explicit
    parameter, that takes the role of the "eps" in LayerNorm, so the network
    doesn't have to do this trick.  We make the "eps" learnable.

    Args:
       num_channels: the number of channels, e.g. 512.
      channel_dim: the axis/dimension corresponding to the channel,
        interprted as an offset from the input's ndim if negative.
        shis is NOT the num_channels; it should typically be one of
        {-2, -1, 0, 1, 2, 3}.
       eps: the initial "epsilon" that we add as ballast in:
             scale = ((input_vec**2).mean() + epsilon)**-0.5
          Note: our epsilon is actually large, but we keep the name
          to indicate the connection with conventional LayerNorm.
       learn_eps: if true, we learn epsilon; if false, we keep it
         at the initial value.
    """

    def __init__(
        self,
        num_channels: int,
        channel_dim: int = -1,  # CAUTION: see documentation.
        eps: float = 0.25,
        learn_eps: bool = True,
    ) -> None:
        super(BasicNorm, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        if learn_eps:
            self.eps = nn.Parameter(torch.tensor(eps).log().detach())
        else:
            self.register_buffer("eps", torch.tensor(eps).log().detach())

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[self.channel_dim] == self.num_channels
        scales = (
            torch.mean(x ** 2, dim=self.channel_dim, keepdim=True)
            + self.eps.exp()
        ) ** -0.5
        return x * scales


class ScaledLinear(nn.Linear):
    """
    A modified version of nn.Linear that gives an easy way to set the
    default initial parameter scale.

    Args:
        Accepts the standard args and kwargs that nn.Linear accepts
        e.g. in_features, out_features, bias=False.

        initial_scale: you can override this if you want to increase
           or decrease the initial magnitude of the module's output
           (affects the initialization of weight_scale and bias_scale).
           Another option, if you want to do something like this, is
           to re-initialize the parameters.
    """

    def __init__(
        self,
        *args,
        initial_scale: float = 1.0,
        **kwargs
    ):
        super(ScaledLinear, self).__init__(*args, **kwargs)
        with torch.no_grad():
            self.weight[:] *= initial_scale
            if self.bias is not None:
                torch.nn.init.uniform_(self.bias,
                                       -0.1 * initial_scale,
                                       0.1 * initial_scale)



class ScaledConv1d(nn.Conv1d):
    # See docs for ScaledLinear
    def __init__(
        self,
        *args,
        initial_scale: float = 1.0,
        **kwargs
    ):
        super(ScaledConv1d, self).__init__(*args, **kwargs)
        with torch.no_grad():
            self.weight[:] *= initial_scale
            if self.bias is not None:
                torch.nn.init.uniform_(self.bias,
                                       -0.1 * initial_scale,
                                       0.1 * initial_scale)

    def get_weight(self):  # TODO: delete
        return self.weight

    def get_bias(self):  # TODO: delete
        return self.bias


class ScaledConv2d(nn.Conv2d):
    # See docs for ScaledLinear
    def __init__(
        self,
        *args,
        initial_scale: float = 1.0,
        **kwargs
    ):
        super(ScaledConv2d, self).__init__(*args, **kwargs)
        with torch.no_grad():
            self.weight[:] *= initial_scale
            if self.bias is not None:
                torch.nn.init.uniform_(self.bias,
                                       -0.1 * initial_scale,
                                       0.1 * initial_scale)

    def get_weight(self):
        return self.weight

    def get_bias(self):
        return  self.bias



class ActivationBalancer(torch.nn.Module):
    """
    Modifies the backpropped derivatives of a function to try to encourage, for
    each channel, that it is positive at least a proportion `threshold` of the
    time.  It does this by multiplying negative derivative values by up to
    (1+max_factor), and positive derivative values by up to (1-max_factor),
    interpolated from 1 at the threshold to those extremal values when none
    of the inputs are positive.


    Args:
           channel_dim: the dimension/axis corresponding to the channel, e.g.
               -1, 0, 1, 2; will be interpreted as an offset from x.ndim if negative.
           min_positive: the minimum, per channel, of the proportion of the time
               that (x > 0), below which we start to modify the derivatives.
           max_positive: the maximum, per channel, of the proportion of the time
               that (x > 0), above which we start to modify the derivatives.
           max_factor: the maximum factor by which we modify the derivatives for
              either the sign constraint or the magnitude constraint;
              e.g. with max_factor=0.02, the the derivatives would be multiplied by
              values in the range [0.98..1.02].
           min_abs:  the minimum average-absolute-value per channel, which
              we allow, before we start to modify the derivatives to prevent
              this.
           max_abs:  the maximum average-absolute-value per channel, which
               we allow, before we start to modify the derivatives to prevent
               this.
    """

    def __init__(
        self,
        channel_dim: int,
        min_positive: float = 0.05,
        max_positive: float = 0.95,
        max_factor: float = 0.01,
        min_abs: float = 0.2,
        max_abs: float = 100.0,
    ):
        super(ActivationBalancer, self).__init__()
        self.channel_dim = channel_dim
        self.min_positive = min_positive
        self.max_positive = max_positive
        self.max_factor = max_factor
        self.min_abs = min_abs
        self.max_abs = max_abs

    def forward(self, x: Tensor) -> Tensor:
        if torch.jit.is_scripting():
            return x

        return ActivationBalancerFunction.apply(
            x,
            self.channel_dim,
            self.min_positive,
            self.max_positive,
            self.max_factor,
            self.min_abs,
            self.max_abs,
        )


class DoubleSwishFunction(torch.autograd.Function):
    """
      double_swish(x) = x * torch.sigmoid(x-1)
    This is a definition, originally motivated by its close numerical
    similarity to swish(swish(x)), where swish(x) =  x * sigmoid(x).

    Memory-efficient derivative computation:
     double_swish(x) = x * s, where s(x) = torch.sigmoid(x-1)
     double_swish'(x) = d/dx double_swish(x) =  x * s'(x) + x' * s(x) = x * s'(x) + s(x).
     Now, s'(x) = s(x) * (1-s(x)).
     double_swish'(x) =  x * s'(x) + s(x).
                      =  x * s(x) * (1-s(x)) + s(x).
                     = double_swish(x) * (1-s(x)) + s(x)
     ... so we just need to remember s(x) but not x itself.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        x = x.detach()
        s = torch.sigmoid(x - 1.0)
        y = x * s
        ctx.save_for_backward(s, y)
        return y

    @staticmethod
    def backward(ctx, y_grad: Tensor) -> Tensor:
        s, y = ctx.saved_tensors
        return (y * (1 - s) + s) * y_grad


class DoubleSwish(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """Return double-swish activation function which is an approximation to Swish(Swish(x)),
        that we approximate closely with x * sigmoid(x-1).
        """
        if torch.jit.is_scripting():
            return x * torch.sigmoid(x - 1.0)
        return DoubleSwishFunction.apply(x)




class GaussProjDrop(torch.nn.Module):
    """
    This has an effect similar to torch.nn.Dropout, but does not privilege the on-axis directions.
    The directions of dropout are fixed when the class is initialized, and are orthogonal.

    dropout_rate: the dropout probability (actually will define the number of zeroed-out directions)
     channel_dim: the axis corresponding to the channel, e.g. -1, 0, 1, 2.
    """
    def __init__(self,
                 num_channels: int,
                 dropout_rate: float = 0.1,
                 channel_dim: int = -1):
        super(GaussProjDrop, self).__init__()
        self.dropout_rate = dropout_rate
        # this formula for rand_scale was found empirically, trying to match the
        # statistics of dropout in terms of cross-correlation with the input, see
        # _test_gauss_proj_drop()
        self.rand_scale = (dropout_rate / (1-dropout_rate)) ** 0.5 # * (num_channels ** -0.5)

        self.channel_dim = channel_dim

        rand_mat = torch.randn(num_channels, num_channels)
        U, _, _ = rand_mat.svd()
        self.register_buffer('U', U)  # a random orthogonal square matrix.  will be a buffer.


    def _randperm_like(self, x: Tensor):
        """
        Returns random permutations of the integers [0,1,..x.shape[-1]-1],
        with the same shape as x.  All dimensions of x other than the last dimension
        will be treated as batch dimensions.

        Torch's randperm does not support a batch dimension, so we pseudo-randomly simulate it.

        For now, requires x.shape[-1] to be either a power of 2 or 3 times a power of 2, as
        we normally set channel dims.  This is required for some number theoretic stuff.
        """
        n = x.shape[-1]

        assert n & (n-1) == 0 or (n//3 & (n//3 - 1)) == 0

        b = x.numel() // n
        randint = random.randint(0, 1000)
        perm = torch.randperm(n, device=x.device)
        # ensure all elements of batch_rand are coprime to n; this will ensure
        # that multiplying the permutation by batch_rand and taking modulo
        # n leaves us with permutations.
        batch_rand = torch.arange(b, device=x.device) * (randint * 6) + 1
        batch_rand = batch_rand.unsqueeze(-1)
        ans = (perm * batch_rand) % n
        ans = ans.reshape(x.shape)
        return ans


    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        else:
            x = x.transpose(self.channel_dim, -1)  # (..., num_channels)
            x_bypass = x # will be used for "+ I"
            perm = self._randperm_like(x)
            x = torch.gather(x, -1, perm)
            # self.U will act like a different matrix for every row of x, because of the random
            # permutation.
            x = torch.matmul(x, self.U)
            x_next = torch.empty_like(x)
            # scatter_ uses perm in opposite way
            # from gather, inverting it.
            x_next.scatter_(-1, perm, x)
            x = (x_next * self.rand_scale + x_bypass)
            return x


def _compute_correlation_loss(cov: Tensor,
                              eps: float) -> Tensor:
    """
    Computes the correlation `loss`, which would be zero if the channel dimensions
    are un-correlated, and equals num_channels if they are maximally correlated
    (i.e., they all point in the same direction)
    Args:
       cov: Uncentered covariance matrix of shape (num_channels, num_channels),
            does not have to be normalized by count.
    """
    inv_sqrt_diag = (cov.diag() + eps) ** -0.5
    norm_cov = cov * (inv_sqrt_diag * inv_sqrt_diag.unsqueeze(-1))
    num_channels = cov.shape[0]
    loss = ((norm_cov - norm_cov.diag().diag()) ** 2).sum() / num_channels
    return loss

def _update_cov_stats(cov: Tensor,
                      x: Tensor,
                      beta: float) -> Tensor:
    """
    Updates covariance stats as a decaying sum, returning the result.
      cov: Old covariance stats, to be added to and returned, of shape
           (num_channels, num_channels)
        x: Tensor of features/activations, of shape (num_frames, num_channels)
     beta: The decay constant for the stats, e.g. 0.8.
    """
    new_cov = torch.matmul(x.t(), x)
    return cov * beta + new_cov * (1-beta)


class DecorrelateFunction(torch.autograd.Function):
    """
    Function object for a function that does nothing in the forward pass;
    but, in the backward pass, adds derivatives that encourage the channel dimensions
    to be un-correlated with each other.
    This should not be used in a half-precision-enabled area, use
    with torch.cuda.amp.autocast(enabled=False).

    Args:
           x: The input tensor, which is also the function output.  It can have
               arbitrary shape, but its dimension `channel_dim` (e.g. -1) will be
               interpreted as the channel dimension.
     old_cov: Covariance statistics from previous frames, accumulated as a decaying
              sum over frames (not average), decaying by beta each time.
       scale: The scale on the derivatives arising from this, expressed as
              a fraction of the norm of the derivative at the output (applied per
              frame).  We will further scale down the derivative if the normalized
              loss is less than 1.
         eps: Epsilon value to prevent division by zero when estimating diagonal
              covariance.
        beta: Decay constant that determines how we combine stats from x with
              stats from cov; e.g. 0.8.
 channel_dim: The dimension/axis of x corresponding to the channel, e.g. 0, 1, 2, -1.
    """
    @staticmethod
    def forward(ctx, x: Tensor, old_cov: Tensor,
                scale: float, eps: float, beta: float,
                channel_dim: int) -> Tensor:
        ctx.save_for_backward(x.detach(), old_cov.detach())
        ctx.scale = scale
        ctx.eps = eps
        ctx.beta = beta
        ctx.channel_dim = channel_dim
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor) -> Tuple[Tensor, None, None, None, None, None]:
        x, old_cov = ctx.saved_tensors

        # Reshape x and x_grad to be (num_frames, num_channels)
        x = x.transpose(-1, ctx.channel_dim)
        x_grad = x_grad.transpose(-1, ctx.channel_dim)
        num_channels = x.shape[-1]
        full_shape = x.shape
        x = x.reshape(-1, num_channels)
        x_grad = x_grad.reshape(-1, num_channels)
        x.requires_grad = True

        # Now, normalize the contributions of frames/pixels x to the covariance,
        # to have magnitudes proportional to the norm of the gradient on that
        # frame; the goal is to exclude "don't-care" frames such as padding frames from
        # the computation.
        x_grad_sqrt_norm = (x_grad ** 2).sum(dim=1) ** 0.25
        x_grad_sqrt_norm /= x_grad_sqrt_norm.mean()
        x_grad_sqrt_norm_is_inf = (x_grad_sqrt_norm - x_grad_sqrt_norm != 0)
        x_grad_sqrt_norm.masked_fill_(x_grad_sqrt_norm_is_inf, 1.0)

        with torch.enable_grad():
            # scale up frames with larger grads.
            x_scaled = x * x_grad_sqrt_norm.unsqueeze(-1)

            cov = _update_cov_stats(old_cov, x_scaled, ctx.beta)
            assert old_cov.dtype != torch.float16
            old_cov[:] = cov  # update the stats outside!  This is not really
                              # how backprop is supposed to work, but this input
                              # is not differentiable..
            loss = _compute_correlation_loss(cov, ctx.eps)
            assert loss.dtype == torch.float32

            if random.random() < 0.05:
                logging.info(f"Decorrelate: loss = {loss}")

            loss.backward()

        decorr_x_grad = x.grad

        scale = ctx.scale * ((x_grad ** 2).mean()  / ((decorr_x_grad ** 2).mean() + 1.0e-20)) ** 0.5
        decorr_x_grad = decorr_x_grad * scale

        x_grad = x_grad + decorr_x_grad

        # reshape back to original shape
        x_grad = x_grad.reshape(full_shape)
        x_grad = x_grad.transpose(-1, ctx.channel_dim)

        return x_grad, None, None, None, None, None



class Decorrelate(torch.nn.Module):
    """
    This module does nothing in the forward pass, but in the backward pass, modifies
    the derivatives in such a way as to encourage the dimensions of its input to become
    decorrelated.

    Args:
            num_channels:  The number of channels, e.g. 256.
          apply_prob_decay:  The probability with which we apply this each time, in
                           training mode, will decay as apply_prob_decay/(apply_prob_decay + step).
                   scale: This number determines the scale of the gradient contribution from
                          this module, relative to whatever the gradient was before;
                          this is applied per frame or pixel, by scaling gradients.
                     eps: An epsilon used to prevent division by zero.
                    beta: A value 0 < beta < 1 that controls decay of covariance stats
             channel_dim: The dimension of the input corresponding to the channel, e.g.
                          -1, 0, 1, 2.

    """
    def __init__(self,
                 num_channels: int,
                 scale: float = 0.1,
                 apply_prob_decay: int = 1000,
                 eps: float = 1.0e-05,
                 beta: float = 0.95,
                 channel_dim: int = -1):
        super(Decorrelate, self).__init__()
        self.scale = scale
        self.apply_prob_decay = apply_prob_decay
        self.eps = eps
        self.beta = beta
        self.channel_dim = channel_dim

        self.register_buffer('cov', torch.zeros(num_channels, num_channels))
        # step_buf is a copy of step, included so it will be loaded/saved with
        # the model.
        self.register_buffer('step_buf', torch.tensor(0.0))
        self.step = 0


    def load_state_dict(self, *args, **kwargs):
        super(Decorrelate, self).load_state_dict(*args, **kwargs)
        self.step = int(self.step_buf.item())


    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        else:
            apply_prob = self.apply_prob_decay / (self.step + self.apply_prob_decay)
            self.step += 1
            self.step_buf.fill_(float(self.step))
            if random.random() > apply_prob:
                return x
            with torch.cuda.amp.autocast(enabled=False):
                x = x.to(torch.float32)
                # the function updates self.cov in its backward pass (it needs the gradient
                # norm, for frame weighting).
                ans = DecorrelateFunction.apply(x, self.cov,
                                                self.scale, self.eps, self.beta,
                                                self.channel_dim)  # == x.
                return ans



def _test_activation_balancer_sign():
    probs = torch.arange(0, 1, 0.01)
    N = 1000
    x = 1.0 * (torch.rand(probs.numel(), N) < probs.unsqueeze(-1))
    x = x.detach()
    x.requires_grad = True
    m = ActivationBalancer(
        channel_dim=0,
        min_positive=0.05,
        max_positive=0.98,
        max_factor=0.2,
        min_abs=0.0,
    )

    y_grad = torch.sign(torch.randn(probs.numel(), N))

    y = m(x)
    y.backward(gradient=y_grad)
    print("_test_activation_balancer_sign: x = ", x)
    print("_test_activation_balancer_sign: y grad = ", y_grad)
    print("_test_activation_balancer_sign: x grad = ", x.grad)


def _test_activation_balancer_magnitude():
    magnitudes = torch.arange(0, 1, 0.01)
    N = 1000
    x = torch.sign(torch.randn(magnitudes.numel(), N)) * magnitudes.unsqueeze(
        -1
    )
    x = x.detach()
    x.requires_grad = True
    m = ActivationBalancer(
        channel_dim=0,
        min_positive=0.0,
        max_positive=1.0,
        max_factor=0.2,
        min_abs=0.2,
        max_abs=0.8,
    )

    y_grad = torch.sign(torch.randn(magnitudes.numel(), N))

    y = m(x)
    y.backward(gradient=y_grad)
    print("_test_activation_balancer_magnitude: x = ", x)
    print("_test_activation_balancer_magnitude: y grad = ", y_grad)
    print("_test_activation_balancer_magnitude: x grad = ", x.grad)


def _test_basic_norm():
    num_channels = 128
    m = BasicNorm(num_channels=num_channels, channel_dim=1)

    x = torch.randn(500, num_channels)

    y = m(x)

    assert y.shape == x.shape
    x_rms = (x ** 2).mean().sqrt()
    y_rms = (y ** 2).mean().sqrt()
    print("x rms = ", x_rms)
    print("y rms = ", y_rms)
    assert y_rms < x_rms
    assert y_rms > 0.5 * x_rms


def _test_double_swish_deriv():
    x = torch.randn(10, 12, dtype=torch.double) * 0.5
    x.requires_grad = True
    m = DoubleSwish()
    torch.autograd.gradcheck(m, x)


def _test_gauss_proj_drop():
    D = 384
    x = torch.randn(30000, D)


    for dropout_rate in [0.2, 0.1, 0.01, 0.05]:
        m1 = torch.nn.Dropout(dropout_rate)
        m2 = GaussProjDrop(D, dropout_rate)
        for mode in ['train', 'eval']:
            y1 = m1(x)
            y2 = m2(x)
            xmag = (x*x).mean()
            y1mag = (y1*y1).mean()
            cross1 = (x*y1).mean()
            y2mag = (y2*y2).mean()
            cross2 = (x*y2).mean()
            print(f"rate={dropout_rate}, mode={mode}, xmag = {xmag}, y1mag = {y1mag}, y2mag = {y2mag}, cross1={cross1}, cross2={cross2}")
            m1.eval()
            m2.eval()


def _test_decorrelate():
    D = 384
    x = torch.randn(30000, D)
    # give it a non-unit covariance.
    m = torch.randn(D, D) * (D ** -0.5)
    _, S, _ = m.svd()
    print("M eigs = ", S[::10])
    x = torch.matmul(x, m)


    # check that class Decorrelate does not crash when running..
    decorrelate = Decorrelate(D)
    x.requires_grad = True
    y = decorrelate(x)
    y.sum().backward()

    decorrelate2 = Decorrelate(D)
    decorrelate2.load_state_dict(decorrelate.state_dict())
    assert decorrelate2.step == decorrelate.step




if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _test_decorrelate()
    _test_gauss_proj_drop()
    _test_activation_balancer_sign()
    _test_activation_balancer_magnitude()
    _test_basic_norm()
    _test_double_swish_deriv()