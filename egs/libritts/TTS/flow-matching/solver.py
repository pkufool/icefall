#!/usr/bin/env python3
# Copyright         2024  Xiaomi Corp.        (authors: Han Zhu)
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

from typing import List, Optional, Union

import torch


class DiffusionModel(torch.nn.Module):
    """A wrapper of diffusion models for inference.
    Args:
        model: The diffusion model.
        prediction: The type of prediction. Can be "v" or "data".
        distill: Whether it is a distillation model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        prediction: str = "v",
        distill: bool = False,
        func_name: str = "forward_audio",
    ):
        super().__init__()
        self.model = model
        assert prediction in ["v", "data"]
        self.prediction = prediction
        self.distill = distill
        self.func_name = func_name
        self.model_func = getattr(self.model, func_name)

    def model_forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        text_condition: torch.Tensor,
        audio_condition: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        guidance_scale: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward function that handles different prediction types (data or v prediction)
            and different model types (normal or distill).
        Args:
            t: The current timestep, a tensor of shape (batch, 1, 1) or a tensor of a single float.
            x: The initial value, with the shape (batch, seq_len, emb_dim).
            text_condition: The text_condition of the diffision model, with the shape (batch, seq_len, emb_dim).
            audio_condition: The audio_condition of the diffision model, with the shape (batch, seq_len, emb_dim).
            padding_mask: The mask for padding; True means masked position, with the shape (batch, seq_len).
            guidance_scale: The scale of classifier-free guidance, a float or a tensor of shape (batch, 1, 1).
        Retrun:
            The prediction with the shape (batch, seq_len, emb_dim).
        """
        if guidance_scale is not None:
            assert (
                self.distill
            ), "guidance_scale is only used as the input of the distillation model."

        if self.distill:
            output = self.model_func(
                t=t,
                xt=x,
                text_condition=text_condition,
                audio_condition=audio_condition,
                padding_mask=padding_mask,
                guidance_scale=guidance_scale,
                **kwargs
            )
        else:
            output = self.model_func(
                t=t,
                xt=x,
                text_condition=text_condition,
                audio_condition=audio_condition,
                padding_mask=padding_mask,
                **kwargs
            )
        if self.prediction == "v":
            return output
        elif self.prediction == "data":
            return x + (1 - t) * output

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        text_condition: torch.Tensor,
        audio_condition: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        guidance_scale: Union[float, torch.Tensor] = 0.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward function that Handles the classifier-free guidance.
        Args:
            t: The current timestep, a tensor of shape (batch, 1, 1) or a tensor of a single float.
            x: The initial value, with the shape (batch, seq_len, emb_dim).
            text_condition: The text_condition of the diffision model, with the shape (batch, seq_len, emb_dim).
            audio_condition: The audio_condition of the diffision model, with the shape (batch, seq_len, emb_dim).
            padding_mask: The mask for padding; True means masked position, with the shape (batch, seq_len).
            guidance_scale: The scale of classifier-free guidance, a float or a tensor of shape (batch, 1, 1).
        Retrun:
            The prediction with the shape (batch, seq_len, emb_dim).
        """
        if not torch.is_tensor(guidance_scale):
            guidance_scale = torch.tensor(
                guidance_scale, dtype=t.dtype, device=t.device
            )
        if self.distill:
            return self.model_forward(
                t=t,
                x=x,
                text_condition=text_condition,
                audio_condition=audio_condition,
                padding_mask=padding_mask,
                guidance_scale=guidance_scale,
                **kwargs
            )

        if (guidance_scale == 0.0).all():
            return self.model_forward(
                t=t,
                x=x,
                text_condition=text_condition,
                audio_condition=audio_condition,
                padding_mask=padding_mask,
                **kwargs
            )
        else:
            if t.dim() != 0:
                t = torch.cat([t] * 2, dim=0)

            x = torch.cat([x] * 2, dim=0)
            padding_mask = torch.cat([padding_mask] * 2, dim=0)

            text_condition = torch.cat(
                [torch.zeros_like(text_condition), text_condition], dim=0
            )
            audio_condition = torch.cat([audio_condition, audio_condition], dim=0)

            data_uncond, data_cond = self.model_forward(
                t=t,
                x=x,
                text_condition=text_condition,
                audio_condition=audio_condition,
                padding_mask=padding_mask,
                **kwargs
            ).chunk(2, dim=0)

            res = (1 + guidance_scale) * data_cond - guidance_scale * data_uncond
            return res


class EulerSolver:
    def __init__(
        self,
        model: torch.nn.Module,
        distill: bool = False,
        func_name: str = "forward_audio",
    ):
        """Construct a Euler Solver
        Args:
            model: The diffusion model.
            distill: Whether it is distillation model.
        """

        self.model = DiffusionModel(
            model, prediction="v", distill=distill, func_name=func_name
        )

    def sample(
        self,
        x: torch.Tensor,
        text_condition: torch.Tensor,
        audio_condition: torch.Tensor,
        padding_mask: torch.Tensor,
        num_step: int = 10,
        guidance_scale: Union[float, torch.Tensor] = 0.0,
        t_start: Union[float, torch.Tensor] = 0.0,
        t_end: Union[float, torch.Tensor] = 1.0,
        t_shift: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute the sample at time `t_end` by Euler Solver.
        Args:
            x: The initial value at time `t_start`, with the shape (batch, seq_len, emb_dim).
            text_condition: The text condition of the diffision mode, with the shape (batch, seq_len, emb_dim).
            audio_condition: The audio condition of the diffision model, with the shape (batch, seq_len, emb_dim).
            padding_mask: The mask for padding; True means masked position, with the shape (batch, seq_len).
            num_step: The number of ODE steps.
            guidance_scale: The scale for classifier-free guidance, which is
                a float or a tensor with the shape (batch, 1, 1).
            t_start: the start timestep in the range of [0, 1],
                which is a float or a tensor with the shape (batch, 1, 1).
            t_end: the end time_step in the range of [0, 1],
                which is a float or a tensor with the shape (batch, 1, 1).
            t_shift: shift the t toward smaller numbers so that the sampling
                will emphasize low SNR region. Should be in the range of (0, 1].
                The shifting will be more significant when the number is smaller.

        Returns:
            The approximated solution at time `t_end`.
        """
        device = x.device

        if torch.is_tensor(t_start) and t_start.dim() > 0:
            timesteps = get_time_steps_batch(
                t_start=t_start,
                t_end=t_end,
                num_step=num_step,
                t_shift=t_shift,
                device=device,
            )
        else:
            timesteps = get_time_steps(
                t_start=t_start,
                t_end=t_end,
                num_step=num_step,
                t_shift=t_shift,
                device=device,
            )
        for step in range(num_step):
            v = self.model(
                t=timesteps[step],
                x=x,
                text_condition=text_condition,
                audio_condition=audio_condition,
                padding_mask=padding_mask,
                guidance_scale=guidance_scale,
                **kwargs
            )
            x = x + v * (timesteps[step + 1] - timesteps[step])
        return x


class DPMSolver:
    """
    DPM-Solver++ (multistep version).
    Adapted from: https://github.com/LuChengTHU/dpm-solver/blob/main/dpm_solver_pytorch.py
    """

    def __init__(
        self,
        model: torch.nn.Module,
        distill: bool = False,
        func_name: str = "forward_audio",
    ):
        """Construct a DPM-Solver."""
        self.model = DiffusionModel(
            model, prediction="data", distill=distill, func_name=func_name
        )

    def dpm_solver_first_update(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        model_s: torch.Tensor,
    ) -> torch.Tensor:
        """
        DPM-Solver-1 (equivalent to DDIM) from time `s` to time `t`.

        Args:
            x: The initial value at time `s`.
            s: The starting time, with the shape (1,).
            t: The ending time, with the shape (1,).
            model_s: The model function evaluated at time `s`.
        Returns:
            x_t: The approximated solution at time `t`.
        """
        lambda_s, lambda_t = self.marginal_lambda(s), self.marginal_lambda(t)
        h = lambda_t - lambda_s
        sigma_s, sigma_t = self.marginal_std(s), self.marginal_std(t)
        alpha_t = self.marginal_alpha(t)
        phi_1 = torch.expm1(-h)
        x_t = sigma_t / sigma_s * x - alpha_t * phi_1 * model_s
        return x_t

    def multistep_dpm_solver_second_update(
        self,
        x: torch.Tensor,
        model_prev_list: List[torch.Tensor],
        t_prev_list: List[torch.Tensor],
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Multistep solver DPM-Solver-2 from time `t_prev_list[-1]` to time `t`.

        Args:
            x: The initial value at time `t_prev_list[-1]`.
            model_prev_list: The previous computed model values.
            t_prev_list: The previous times, each time has the shape (1,)
            t: The ending time, with the shape (1,).
        Returns:
            x_t: The approximated solution at time `t`.
        """
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        lambda_prev_1, lambda_prev_0, lambda_t = (
            self.marginal_lambda(t_prev_1),
            self.marginal_lambda(t_prev_0),
            self.marginal_lambda(t),
        )
        sigma_prev_0, sigma_t = self.marginal_std(t_prev_0), self.marginal_std(t)
        alpha_t = self.marginal_alpha(t)

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = (1.0 / r0) * (model_prev_0 - model_prev_1)
        phi_1 = torch.expm1(-h)
        x_t = (
            (sigma_t / sigma_prev_0) * x
            - (alpha_t * phi_1) * model_prev_0
            - 0.5 * (alpha_t * phi_1) * D1_0
        )
        return x_t

    def multistep_dpm_solver_update(
        self,
        x: torch.Tensor,
        model_prev_list: List[torch.Tensor],
        t_prev_list: List[torch.Tensor],
        t: torch.Tensor,
        order: int,
    ) -> torch.Tensor:
        """
        Multistep DPM-Solver with the order `order` from time `t_prev_list[-1]` to time `t`.

        Args:
            x: The initial value at time `t_prev_list[-1]`.
            model_prev_list: The previous computed model values.
            t_prev_list: The previous times, each time has the shape (1,)
            t: The ending time, with the shape (1,).
            order: The order of DPM-Solver. We only support order == 1 or 2.
        Returns:
            x_t: The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(
                x, t_prev_list[-1], t, model_s=model_prev_list[-1]
            )
        elif order == 2:
            return self.multistep_dpm_solver_second_update(
                x, model_prev_list, t_prev_list, t
            )
        else:
            raise ValueError("Solver order must be 1 or 2, got {}".format(order))

    def sample(
        self,
        x: torch.Tensor,
        text_condition: torch.Tensor,
        audio_condition: torch.Tensor,
        padding_mask: torch.Tensor,
        num_step: int = 10,
        guidance_scale: Union[float, torch.Tensor] = 0.0,
        t_start: Union[float, torch.Tensor] = 0.0,
        t_end: Union[float, torch.Tensor] = 1.0,
        t_shift: float = 1.0,
        order: int = 2,
        lower_order_final: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute the sample at time `t_end` by Euler Solver.
        Args:
            x: The initial value at time `t_start`, with the shape (batch, seq_len, emb_dim).
            text_condition: The text condition of the diffision mode, with the shape (batch, seq_len, emb_dim).
            audio_condition: The audio condition of the diffision model, with the shape (batch, seq_len, emb_dim).
            padding_mask: The mask for padding; True means masked position, with the shape (batch, seq_len).
            num_step: The number of ODE steps.
            guidance_scale: The scale for classifier-free guidance, which is
                a float or a tensor with the shape (batch, 1, 1).
            t_start: the start timestep in the range of [0, 1],
                which is a float or a tensor with the shape (batch, 1, 1).
            t_end: the end time_step in the range of [0, 1],
                which is a float or a tensor with the shape (batch, 1, 1).
            t_shift: shift the t toward smaller numbers so that the sampling
                will emphasize low SNR region. Should be in the range of (0, 1].
                The shifting will be more significant when the number is smaller.
            order: the order of DPM-Solver++. 2 is recommended.
            lower_order_final: Whether to use lower order solvers at the final steps.
                This trick is a key to stabilizing the sampling by DPM-Solver with very few steps
                (especially for steps <= 10). So we recommend to set it to be `True`.

        Returns:
            The approximated solution at time `t_end`.
        """
        device = x.device
        assert num_step >= order
        if torch.is_tensor(t_start) and t_start.dim() > 0:
            timesteps = get_time_steps_batch(
                t_start=t_start,
                t_end=t_end,
                num_step=num_step,
                t_shift=t_shift,
                device=device,
            )
        else:
            timesteps = get_time_steps(
                t_start=t_start,
                t_end=t_end,
                num_step=num_step,
                t_shift=t_shift,
                device=device,
            )
        assert timesteps.shape[0] - 1 == num_step
        # Init the initial values.
        step = 0
        t = timesteps[step]
        t_prev_list = [t]
        model_prev_list = [
            self.model(
                t=t,
                x=x,
                text_condition=text_condition,
                audio_condition=audio_condition,
                padding_mask=padding_mask,
                guidance_scale=guidance_scale,
                **kwargs
            )
        ]
        # Init the first `order` values by lower order multistep DPM-Solver.
        for step in range(1, order):
            t = timesteps[step]
            x = self.multistep_dpm_solver_update(
                x, model_prev_list, t_prev_list, t, step
            )
            t_prev_list.append(t)
            model_prev_list.append(
                self.model(
                    t=t,
                    x=x,
                    text_condition=text_condition,
                    audio_condition=audio_condition,
                    padding_mask=padding_mask,
                    guidance_scale=guidance_scale,
                    **kwargs
                )
            )
        # Compute the remaining values by `order`-th order multistep DPM-Solver.
        for step in range(order, num_step + 1):
            t = timesteps[step]
            if lower_order_final:
                step_order = min(order, num_step + 1 - step)
            else:
                step_order = order
            x = self.multistep_dpm_solver_update(
                x, model_prev_list, t_prev_list, t, step_order
            )
            for i in range(order - 1):
                t_prev_list[i] = t_prev_list[i + 1]
                model_prev_list[i] = model_prev_list[i + 1]
            t_prev_list[-1] = t
            # We do not need to evaluate the final model value.
            if step < num_step:
                model_prev_list[-1] = self.model(
                    t=t,
                    x=x,
                    text_condition=text_condition,
                    audio_condition=audio_condition,
                    padding_mask=padding_mask,
                    guidance_scale=guidance_scale,
                    **kwargs
                )
        return x

    def marginal_alpha(self, t: float) -> float:
        """
        Compute alpha_t of a given continuous-time label t.
        """
        return t

    def marginal_std(self, t: float) -> float:
        """
        Compute sigma_t of a given continuous-time label t.
        """
        return 1 - t

    def marginal_lambda(self, t: float) -> float:
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t.
        """
        return torch.log(t) - torch.log(1 - t)


def get_time_steps(
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_step: int = 10,
    t_shift: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Compute the intermediate time steps for sampling.

    Args:
        t_start: The starting time of the sampling (default is 0).
        t_end: The starting time of the sampling (default is 1).
        num_step: The number of sampling.
        t_shift: shift the t toward smaller numbers so that the sampling
            will emphasize low SNR region. Should be in the range of (0, 1].
            The shifting will be more significant when the number is smaller.
        device: A torch device.
    Returns:
        The time step with the shape (num_step + 1,).
    """

    timesteps = torch.linspace(t_start, t_end, num_step + 1).to(device)

    timesteps = t_shift * timesteps / (1 + (t_shift - 1) * timesteps)

    return timesteps


def get_time_steps_batch(
    t_start: torch.Tensor,
    t_end: torch.Tensor,
    num_step: int = 10,
    t_shift: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Compute the intermediate time steps for sampling in the batch mode.

    Args:
        t_start: The starting time of the sampling (default is 0), with the shape (batch, 1, 1).
        t_end: The starting time of the sampling (default is 1), with the shape (batch, 1, 1).
        num_step: The number of sampling.
        t_shift: shift the t toward smaller numbers so that the sampling
            will emphasize low SNR region. Should be in the range of (0, 1].
            The shifting will be more significant when the number is smaller.
        device: A torch device.
    Returns:
        The time step with the shape (num_step + 1, N, 1, 1).
    """
    while t_start.dim() > 1 and t_start.size(-1) == 1:
        t_start = t_start.squeeze(-1)
    while t_end.dim() > 1 and t_end.size(-1) == 1:
        t_end = t_end.squeeze(-1)
    assert t_start.dim() == t_end.dim() == 1

    timesteps_shape = (num_step + 1, t_start.size(0))
    timesteps = torch.zeros(timesteps_shape, device=device)

    for i in range(t_start.size(0)):
        timesteps[:, i] = torch.linspace(t_start[i], t_end[i], steps=num_step + 1)

    timesteps = t_shift * timesteps / (1 + (t_shift - 1) * timesteps)

    return timesteps.unsqueeze(-1).unsqueeze(-1)
