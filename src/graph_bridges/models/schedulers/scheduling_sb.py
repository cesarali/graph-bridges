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

# DIFFUSERS IMPORT ---------------------------------------------------------------------------------
import copy

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, randn_tensor
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin

# GRAPH BRIDGES IMPORT ----------------------------------------------------------------------------

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import numpy as np

from diffusers.utils import BaseOutput, randn_tensor
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin

from graph_bridges.models.schedulers.scheduling_utils import register_scheduler
from graph_bridges.configs.graphs.lobster.config_base import BridgeConfig
from graph_bridges.models.backward_rates.backward_rate import BackwardRate
from graph_bridges.models.reference_process.ctdd_reference import ReferenceProcess
from torch.distributions.poisson import Poisson
from graph_bridges.data.graph_dataloaders import SpinsToBinaryTensor
from graph_bridges.data.graph_dataloaders import BinaryTensorToSpinsTransform

@dataclass
class SBSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.

    Args:
        new_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample (x_{0}) based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    new_sample: torch.FloatTensor
    original_sample: Optional[torch.FloatTensor] = None

@register_scheduler
class SBScheduler(SchedulerMixin, ConfigMixin):
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

    def set_timesteps(
            self,
            num_steps: Optional[int] = None,
            min_t: Optional[float] = None,
            sinkhorn_iteration: int = 0,
            timesteps: Optional[List[float]] = None,
            device: Union[str, torch.device] = None,
    ):
        """
        """
        if num_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `custom_timesteps`.")

        self.min_t = min_t
        self.num_steps = num_steps
        self.timesteps = np.concatenate((np.linspace(1.0, self.min_t, self.num_steps), np.array([0])))
        if sinkhorn_iteration % 2 == 0:
            self.timesteps = self.timesteps[::-1]
        self.timesteps = torch.from_numpy(self.timesteps.copy()).to(device)

        return self.timesteps

    def __len__(self):
        return self.config.num_train_timesteps

    def to(self, device):
        self.device = device
        return self

    def step_poisson(
            self,
            rates_ : torch.FloatTensor,
            x: torch.FloatTensor,
            h: float,
            device: torch.device = None,
            return_dict: bool = True,
    ) -> Union[SBSchedulerOutput, Tuple]:
        """
        """
        assert x.min() == -1. #make sure that we receive spins
        x_new = copy.deepcopy(x)
        poisson_probabilities = rates_ * h

        assert (poisson_probabilities > 0.).all()

        events = Poisson(poisson_probabilities).sample()
        where_to_flip = torch.where(events > 0)

        # Flip accordingly
        x_new[where_to_flip] = x_new[where_to_flip] * (-1.)

        if return_dict:
            output = SBSchedulerOutput(new_sample=x_new, original_sample=x)
            return output
        else:
            return (x_new, x)

    def step_tau(
        self,
        rates_: torch.FloatTensor,
        spins: torch.FloatTensor,
        h:float,
        device:torch.device = None,
        return_dict: bool = True,
    ) -> Union[SBSchedulerOutput, Tuple]:
        """
        """
        assert spins.min() == -1. #make sure that we receive spins
        transforms = SpinsToBinaryTensor()
        x = transforms(spins)
        num_of_paths = x.shape[0]
        diffs = torch.arange(self.S, device=device).view(1, 1, self.S) - x.view(num_of_paths, self.D, 1)
        rates_ = rates_[:, :, None].repeat(1, 1, 2) # here the rates to flip per spins are the same
        poisson_dist = torch.distributions.poisson.Poisson(rates_ * h)
        jump_nums = poisson_dist.sample()
        adj_diffs = jump_nums * diffs
        overall_jump = torch.sum(adj_diffs, dim=2)
        xp = x + overall_jump
        x_new = torch.clamp(xp, min=0, max=self.S - 1)
        new_spins = BinaryTensorToSpinsTransform(x_new)

        if return_dict:
            return SBSchedulerOutput(new_sample=new_spins,
                                     original_sample=spins)
        else:
            return (new_spins, spins)

    def step(self,rates_,x,timestep,h,device,return_dict=True,step_type=None):
        if step_type is None:
            step_type = self.cfg.sampler.name
        if step_type == "TauLeaping":
            return self.step_tau(rates_,x,h,device,return_dict)
        else:
            return self.step_poisson(rates_,x,h,device,return_dict)

    def add_noise(
            self,
            original_samples: torch.FloatTensor,
            reference_process: ReferenceProcess,
            timesteps: torch.IntTensor,
            device: torch.device,
            return_dict: bool = True,

    ) -> Tuple:
        """
        :param original_samples:
        :param reference_process:
        :param timesteps:
        :param device:
        :return:
        """
        return None

    #=====================================
    # MY FUNCTIONS
    #=====================================
    def time_direction(self, forward=True):
        """
        """
        if forward:
            current_time_index = 0
            time_sign = 1
            end_condition = lambda current_index: current_index < self.number_of_time_steps
        else:
            current_time_index = self.number_of_time_steps + 1
            time_sign = -1
            end_condition = lambda current_index: current_index > 1.

        return current_time_index, time_sign, end_condition

    def set_schrodinger_variables(self,
                                  current_model: BackwardRate,
                                  past_to_train_model: BackwardRate,
                                  sinkhorn_iteration=0,
                                  swap_models=False,
                                  restart_current=True):
        """
        """
        if sinkhorn_iteration % 2 == 0:
            if sinkhorn_iteration == 0:
                is_past_forward = True
                reference = True
            else:
                is_past_forward = True
                reference = False
        else:
            is_past_forward = False
            reference = False

        return is_past_forward, reference
