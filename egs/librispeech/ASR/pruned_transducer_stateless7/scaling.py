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
from typing import Optional, Tuple, Union
from functools import reduce
import logging

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class StructuredLinear(torch.nn.Module):
    """
    This module mostly behaves like nn.Linear, but the in_features and out_features
    (the number of input and output channels) are specified as tuples; the
    actual numbers of channels are products over these tuples.
    E.g. (2, 256) means 512, with the slowest-varying/largest-stride dims first
    in terms of the layout.
    For purposes of the forward() function it will behave the same as if the dim
    was 512, but the parameter tensors have this structure, which makes
    a difference if you are using the NeutralGradient optimizer and perhaps
    certain other optimizers.

    Args:
        in_features: The number of input channels, specified as
           a tuple of ints (the number of input channels will be their
           product).  The only difference this makes is that the
           nn.Parameter tensor will be shaped differently, which may
           affect some optimizers.
        out_features: The number of output channels, specified as
          a tuple of ints.
        initial_scale: The default initial parameter scale will be
           multiplied by this.
        bias: If true, include the bias term.
    """
    def __init__(self,
                 in_features: Tuple[int],
                 out_features: Tuple[int],
                 bias: bool = True,
                 initial_scale: float = 1.0) -> None:
        super(StructuredLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        in_size = reduce((lambda i,j: i*j), in_features)
        out_size = reduce((lambda i,j: i*j), out_features)
        self.weight_shape = (out_size, in_size)
        self.weight = nn.Parameter(torch.Tensor(*out_features, *in_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(*out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(initial_scale)


    def reset_parameters(self, initial_scale: float = 1.0) -> None:
        nn.init.kaiming_uniform_(self.weight.reshape(*self.weight_shape), a=(5 ** 0.5))
        with torch.no_grad():
            self.weight *= initial_scale
        nn.init.uniform_(self.bias,
                         -0.1 * initial_scale,
                         0.1 * initial_scale)

    def get_weight(self) -> Tensor:
        return self.weight.reshape(*self.weight_shape)

    def get_bias(self) -> Optional[Tensor]:
        return (None if self.bias is None else
                self.bias.reshape(self.weight_shape[0]))

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.get_weight(), self.get_bias())

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class StructuredConv1d(nn.Conv1d):
    """
    This module mostly behaves like nn.Conv1d, but the
    in_channels and out_channels are specified as tuples.  For example,
    512 channels might be specified as
    (2, 256), with slowest-varying/largest-stride dims first in terms of the layout.
    For purposes of the forward() function it will behave the same as if the dim
    was 512, but the parameter tensors have this structure, which makes
    a difference if you are using the NeutralGradient optimizer.


    Args:
        in_channels: The number of input channels, specified as
           a tuple of ints (the number of input channels will be their
           product).  The only difference this makes is that the
           nn.Parameter tensor will be shaped differently, which may
           affect some optimizers.
        out_channels: The number of output channels, specified as
          a tuple of ints.
        initial_scale: The default initial parameter scale will be
           multiplied by this.
        bias: If true, include the bias term.
    """
    def __init__(
        self,
        in_channels: Tuple[int],
        out_channels: Tuple[int],
        *args,
        initial_scale: float = 1.0,
        **kwargs
    ):
        super(StructuredConv1d, self).__init__(
            reduce((lambda i,j: i*j), in_channels),
            reduce((lambda i,j: i*j), out_channels),
            *args, **kwargs)

        assert self.groups == 1, "Groups not supported as yet"

        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.transposed:
            in_channels, out_channels = out_channels, in_channels

        self.weight_shape = self.weight.shape
        self.weight = nn.Parameter(self.weight.detach().reshape(
            *out_channels, *in_channels, *self.weight.shape[2:]))

        self.bias_shape = self.bias.shape
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.detach().reshape(
                *out_channels))

        # These changes in the initialization are the same as for class ScaledConv1d.
        with torch.no_grad():
            self.weight[:] *= initial_scale
            if self.bias is not None:
                torch.nn.init.uniform_(self.bias,
                                       -0.1 * initial_scale,
                                       0.1 * initial_scale)

    def get_weight(self) -> Tensor:
        return self.weight.reshape(*self.weight_shape)
    def get_bias(self) -> Optional[Tensor]:
        return (None if self.bias is None else
                self.bias.reshape(*self.bias_shape))

    def forward(self, input: Tensor) -> Tensor:
        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            self.get_weight(), self.get_bias(), self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv1d(input, self.get_weight(), self.get_bias(), self.stride,
                        self.padding, self.dilation, self.groups)





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

def _test_structured_linear():
    m = StructuredLinear((2, 100), (3, 100), bias=True)
    assert m.weight.shape == (3, 100, 2, 100)
    assert m.bias.shape == (3, 100)
    x = torch.randn(50, 200)
    y = m(x)
    assert y.shape == (50, 300)

def _test_structured_conv1d():
    m = StructuredConv1d((2, 100), (3, 100), kernel_size=3, padding=1, bias=True)
    assert m.weight.shape == (3, 100, 2, 100, 3)
    assert m.bias.shape == (3, 100)
    T = 39
    x = torch.randn(50, 200, T)
    y = m(x)
    assert y.shape == (50, 300, T)

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



if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _test_structured_linear()
    _test_structured_conv1d()
    _test_gauss_proj_drop()
    _test_activation_balancer_sign()
    _test_activation_balancer_magnitude()
    _test_basic_norm()
    _test_double_swish_deriv()