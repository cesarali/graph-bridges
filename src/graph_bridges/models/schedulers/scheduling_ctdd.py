# Copyright 2023 UC Berkeley Team and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This file is strongly influenced by https://github.com/ermongroup/ddim

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.functional as F


from diffusers.utils import BaseOutput, randn_tensor
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin

from graph_bridges.models.reference_process.ctdd_reference import ReferenceProcess
from graph_bridges.models.schedulers.scheduling_utils import register_scheduler
from graph_bridges.configs.graphs.lobster.config_base import BridgeConfig
from graph_bridges.models.backward_rates.backward_rate import GaussianTargetRateImageX0PredEMA
from graph_bridges.data.dataloaders import DoucetTargetData, GraphSpinsDataLoader
from graph_bridges.models.samplers.sampling import ReferenceProcess
from graph_bridges.models.reference_process.ctdd_reference import ReferenceProcess
from graph_bridges.models.losses.ctdd_losses import GenericAux
from graph_bridges.data.dataloaders_config import GraphSpinsDataLoaderConfig



@dataclass
class CTDDSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample (x_{0}) based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None

@dataclass
class CTDDSchedulerNoiseOutput(BaseOutput):
    """
    Output class for the scheduler's add noise output.

    """
    #loss.add_noise(minibatch, model, ts, device)
    x_t : torch.FloatTensor
    x_tilde : torch.FloatTensor
    qt0 : torch.FloatTensor
    rate : torch.FloatTensor


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


@register_scheduler
class CTDDScheduler(SchedulerMixin, ConfigMixin):
    """

    """
    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        config:BridgeConfig,
        device:torch.device,
        num_train_timesteps: int = 1000,
    ):
        self.cfg = config
        self.S = self.cfg.data.S
        self.D = self.cfg.data.D
        self.device = device
        print("Scheduler")

    def set_timesteps(
        self,
        num_steps: Optional[int] = None,
        min_t: Optional[float] = None,
        timesteps:Optional[List[float]] = None,
        device: Union[str, torch.device] = None,
    ):
        """
        """
        if num_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `custom_timesteps`.")

        self.min_t = min_t
        self.num_steps = num_steps
        self.timesteps = np.concatenate((np.linspace(1.0, self.min_t, self.num_steps), np.array([0])))
        self.timesteps = torch.from_numpy(self.timesteps).to(device)

    def __len__(self):
        return self.config.num_train_timesteps

    def to(self,device):
        self.device = device
        return self

    def step(
        self,
        rates_: torch.FloatTensor,
        x: torch.FloatTensor,
        timestep: int,
        h:float,
        device:torch.device = None,
        return_dict: bool = True,
    ) -> Union[CTDDSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            generator: random number generator.
            return_dict (`bool`): option for returning tuple rather than DDPMSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.DDPMSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.DDPMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        num_of_paths = x.shape[0]
        diffs = torch.arange(self.S, device=device).view(1, 1, self.S) - x.view(num_of_paths, self.D, 1)
        poisson_dist = torch.distributions.poisson.Poisson(rates_ * h)
        jump_nums = poisson_dist.sample()
        adj_diffs = jump_nums * diffs
        overall_jump = torch.sum(adj_diffs, dim=2)
        xp = x + overall_jump
        x_new = torch.clamp(xp, min=0, max=self.S - 1)

        if return_dict:
            return CTDDSchedulerOutput(prev_sample=x_new,
                                       pred_original_sample=x)
        else:
            return (x_new,x)

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        reference_process: ReferenceProcess,
        timesteps: torch.IntTensor,
        device:torch.device,
        return_dict: bool = True,

    ) -> Union[CTDDSchedulerNoiseOutput, Tuple]:
        """
            :param original_samples:
            :param reference_process:
            :param timesteps:
            :param device:
            :return:

            Returns:
            [`~schedulers.scheduling_utils.DDPMSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.DDPMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        S = self.cfg.data.S
        if len(original_samples.shape) == 4:
            B, C, H, W = original_samples.shape
            minibatch = original_samples.view(B, C * H * W)
        B, D = minibatch.shape

        qt0 = reference_process.transition(timesteps)  # (B, S, S)
        rate = reference_process.rate(timesteps)  # (B, S, S)

        # --------------- Sampling x_t, x_tilde --------------------

        qt0_rows_reg = qt0[
                       torch.arange(B, device=device).repeat_interleave(D),
                       minibatch.flatten().long(),
                       :
                       ]  # (B*D, S)

        x_t_cat = torch.distributions.categorical.Categorical(qt0_rows_reg)
        x_t = x_t_cat.sample().view(B, D)

        rate_vals_square = rate[
                           torch.arange(B, device=device).repeat_interleave(D),
                           x_t.long().flatten(),
                           :
                           ]  # (B*D, S)
        rate_vals_square[
            torch.arange(B * D, device=device),
            x_t.long().flatten()
        ] = 0.0  # 0 the diagonals
        rate_vals_square = rate_vals_square.view(B, D, S)
        rate_vals_square_dimsum = torch.sum(rate_vals_square, dim=2).view(B, D)
        square_dimcat = torch.distributions.categorical.Categorical(
            rate_vals_square_dimsum
        )
        square_dims = square_dimcat.sample()  # (B,) taking values in [0, D)
        rate_new_val_probs = rate_vals_square[
                             torch.arange(B, device=device),
                             square_dims,
                             :
                             ]  # (B, S)
        square_newvalcat = torch.distributions.categorical.Categorical(
            rate_new_val_probs
        )
        square_newval_samples = square_newvalcat.sample()  # (B, ) taking values in [0, S)
        x_tilde = x_t.clone()
        x_tilde[
            torch.arange(B, device=device),
            square_dims
        ] = square_newval_samples

        if not return_dict:
            return (x_t,x_tilde,qt0,rate)

        return CTDDSchedulerNoiseOutput(x_t=x_t,
                                        x_tilde=x_tilde,
                                        qt0=qt0,
                                        rate=rate)

    def previous_timestep(self, timestep):
        if self.custom_timesteps:
            index = (self.timesteps == timestep).nonzero(as_tuple=True)[0][0]
            if index == self.timesteps.shape[0] - 1:
                prev_t = torch.tensor(-1)
            else:
                prev_t = self.timesteps[index + 1]
        else:
            num_inference_steps = (
                self.num_inference_steps if self.num_inference_steps else self.config.num_train_timesteps
            )
            prev_t = timestep - self.config.num_train_timesteps // num_inference_steps

        return prev_t


if __name__ == "__main__":
    from graph_bridges.models.schedulers.scheduling_utils import create_scheduler
    from graph_bridges.models.backward_rates.backward_rate_utils import create_model
    from graph_bridges.models.reference_process.reference_process_utils import create_reference
    from graph_bridges.data.dataloaders_utils import create_dataloader
    from graph_bridges.models.losses.loss_utils import create_loss
    from pprint import pprint

    from graph_bridges.configs.graphs.lobster.config_base import BridgeConfig, get_config_from_file

    config = get_config_from_file("graph","lobster","1687884918")
    #pprint(config)

    device = torch.device(config.device)
    # =================================================================
    # CREATE OBJECTS FROM CONFIGURATION

    data_dataloader: GraphSpinsDataLoader
    model: GaussianTargetRateImageX0PredEMA
    reference_process: ReferenceProcess
    loss: GenericAux
    scheduler: CTDDScheduler

    data_dataloader = create_dataloader(config, device)
    model = create_model(config, device)
    reference_process = create_reference(config, device)
    loss = create_loss(config, device)
    scheduler = create_scheduler(config, device)

    databatch = next(data_dataloader.train().__iter__())[0]
    print(databatch.shape)
    """
    scheduler.step(model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        return_dict: bool = True,
        device:torch.device = None,)
    """