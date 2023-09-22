# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from typing import Optional, Tuple, Union
import torch

from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from graph_bridges.configs.graphs.graph_config_sb import SBConfig
import torch.nn.functional as F
from tqdm import tqdm
from graph_bridges.models.schedulers.scheduling_sb import SBScheduler
from graph_bridges.models.reference_process.ctdd_reference import ReferenceProcess
from graph_bridges.models.backward_rates.ctdd_backward_rate import BackwardRate
from graph_bridges.data.graph_dataloaders import BridgeGraphDataLoaders

class SBPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """
    config : SBConfig
    model: BackwardRate
    reference_process: ReferenceProcess
    data: BridgeGraphDataLoaders
    target: BridgeGraphDataLoaders
    scheduler: SBScheduler

    def __init__(self,
                 config:SBConfig,
                 reference_process:ReferenceProcess,
                 data:BridgeGraphDataLoaders,
                 target:BridgeGraphDataLoaders,
                 scheduler:SBScheduler):

        super().__init__()
        self.register_modules(reference_process=reference_process,
                              data=data,
                              target=target,
                              scheduler=scheduler)
        self.bridge_config = config
        self.D = self.bridge_config.data.D


    def select_time_difference(self,sinkhorn_iteration,timesteps,idx):
        if sinkhorn_iteration % 2 == 0:
            h = timesteps[idx + 1] - timesteps[idx]
        else:
            h = timesteps[idx] - timesteps[idx + 1]
        return h

    def select_data_iterator(self,sinkhorn_iteration,train):
        """

        :param sinkhorn_iteration:
        :param train:
        :return: data_iterator
        """
        # Sample gaussian noise to begin loop
        if sinkhorn_iteration % 2 == 0:
            if train:
                data_iterator = self.data.train().__iter__()
            else:
                data_iterator = self.data.test().__iter__()
        else:
            if train:
                data_iterator = self.target.train().__iter__()
            else:
                data_iterator = self.target.test().__iter__()
        return data_iterator

    def select_data_sampler(self,sinkhorn_iteration,train):
        """

        :param sinkhorn_iteration:
        :param train:
        :return: data_iterator
        """
        # Sample gaussian noise to begin loop
        if sinkhorn_iteration % 2 == 0:
            if train:
                data_sampler = self.data
            else:
                data_sampler = self.data
        else:
            if train:
                data_sampler = self.target
            else:
                data_sampler = self.target
        return data_sampler

    def paths_shapes(self,full_path,timesteps,return_path_shape):
        if isinstance(full_path,list):
            full_path = torch.concat(full_path, dim=1)

        number_of_paths = full_path.shape[0]
        number_of_timesteps = full_path.shape[1]

        timesteps = timesteps.unsqueeze(0).repeat(number_of_paths, 1)
        if return_path_shape:
            return full_path, timesteps
        else:
            timesteps = timesteps.reshape(number_of_paths * number_of_timesteps)
            full_path = full_path.reshape(number_of_paths * number_of_timesteps, -1)
            return full_path, timesteps

    def paths_iterator(self,
                       generation_model: Union[BackwardRate,ReferenceProcess] = None,
                       sinkhorn_iteration = 0,
                       device :torch.device = None,
                       train: bool = True,
                       return_dict: bool = True,
                       return_path:bool = True,
                       return_path_shape:bool = False):

        if generation_model is None:
            assert sinkhorn_iteration == 0
        else:
            if isinstance(generation_model, BackwardRate):
                assert sinkhorn_iteration >= 1

        # set step values
        self.scheduler.set_timesteps(self.bridge_config.sampler.num_steps,
                                     self.bridge_config.sampler.min_t,
                                     sinkhorn_iteration=sinkhorn_iteration)
        timesteps_ = self.scheduler.timesteps
        data_iterator = self.select_data_iterator(sinkhorn_iteration, train)

        for databatch in data_iterator:
            spins = databatch[0]
            num_of_paths = spins.shape[0]
            spins = spins.to(device)
            if return_path:
                full_path = [spins.unsqueeze(1)]

            for idx, t in enumerate(timesteps_[0:-1]):

                h = self.select_time_difference(sinkhorn_iteration,timesteps_,idx)
                times_ = t * torch.ones(num_of_paths,device=device)

                if sinkhorn_iteration != 0:
                    logits = generation_model.stein_binary_forward(spins, times_)
                    rates_ = F.softplus(logits)
                else:
                    rates_ = self.reference_process.rates_states_and_times(spins,times_)

                spins_new = self.scheduler.step(rates_,spins,t,h,device=device,return_dict=True,step_type="Poisson").new_sample
                spins = spins_new

                if return_path:
                    full_path.append(spins.unsqueeze(1))

            if return_path:
                full_path = torch.concat(full_path, dim=1)
                timesteps = timesteps_.unsqueeze(0).repeat(num_of_paths,1)

                if return_path_shape:
                    yield (full_path, timesteps)
                else:
                    number_of_paths = full_path.shape[0]
                    number_of_timesteps = full_path.shape[1]

                    timesteps = timesteps.reshape(number_of_paths * number_of_timesteps)
                    full_path = full_path.reshape(number_of_paths * number_of_timesteps, -1)

                    yield (full_path, timesteps)
            else:
                yield spins_new

    @torch.no_grad()
    def __call__(
        self,
        generation_model: Optional[BackwardRate] = None,
        sinkhorn_iteration = 0,
        device :torch.device = None,
        initial_spins: torch.Tensor = None,
        train: bool =True,
        sample_from_reference_native: bool = True,
        sample_size: int = None,
        return_dict: bool = True,
        return_path:bool = True,
        return_path_shape:bool = False,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """

        :param generation_model:
        :param sinkhorn_iteration:
        :param device:
        :param x:
        :param train:
        :param sample_from_reference_native: use the native sampling from the reference
        :param return_dict:
        :param return_path: bool  gives you the whole path, otherwise is set to zero
        :param return_path_shape: returns paths as [batchsize,num_timesteps,dimension]

        :return:
        """
        #=========================================================
        # PREPROCESSING
        #=========================================================

        # reference process is only available for sinkhorn iteration 0
        if generation_model is None:
            assert sinkhorn_iteration == 0
        else:
            if isinstance(generation_model, BackwardRate):
                assert sinkhorn_iteration >= 1

        # set step values
        self.scheduler.set_timesteps(self.bridge_config.sampler.num_steps,
                                     self.bridge_config.sampler.min_t,
                                     sinkhorn_iteration=sinkhorn_iteration)
        timesteps = self.scheduler.timesteps
        timesteps = timesteps.to(device)

        # sample initial state
        if initial_spins is None:
            if sample_size is None:
                sample_size = self.bridge_config.data.batch_size
            data_sampler = self.select_data_sampler(sinkhorn_iteration,train)
            initial_spins = data_sampler.sample(sample_size)[0]

        # preprocess initial state and full path
        initial_spins = initial_spins.to(device)
        num_of_paths = initial_spins.shape[0]
        if return_path:
            full_path = [initial_spins.unsqueeze(1)]

        # =============================================================
        # SAMPLE NATIVELY
        # ==============================================================
        if sinkhorn_iteration == 0 and sample_from_reference_native and hasattr(self.reference_process,'sample_path'):
            full_path, timesteps = self.reference_process.sample_path(initial_spins, timesteps)
            if return_path:
                return self.paths_shapes(full_path, timesteps, return_path_shape)
            else:
                return full_path[:,-1,:]
        else:
            #=========================================================
            # SAMPLING LOOP
            #=========================================================
            for idx, t in tqdm(enumerate(timesteps[0:-1])):

                h = self.select_time_difference(sinkhorn_iteration, timesteps, idx)
                times = t * torch.ones(num_of_paths,device=device)

                if sinkhorn_iteration != 0:
                    logits = generation_model.stein_binary_forward(initial_spins, times)
                    rates_ = F.softplus(logits)
                else:
                    rates_ = self.reference_process.rates_states_and_times(initial_spins, times)

                spins_new = self.scheduler.step(rates_,
                                                initial_spins,
                                                t,
                                                h,
                                                device,
                                                return_dict=True,
                                                step_type=self.bridge_config.sampler.step_type).new_sample
                initial_spins = spins_new

                if return_path:
                    full_path.append(initial_spins.unsqueeze(1))

            #=========================================================
            # HANDLES PATH SHAPES
            #=========================================================
            if return_path:
                return self.paths_shapes(full_path,timesteps,return_path_shape)
            else:
                return spins_new