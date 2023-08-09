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
from typing import List, Optional, Tuple, Union
import torch

from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from graph_bridges.configs.graphs.config_sb import SBConfig
from diffusers.utils import randn_tensor
import torch.nn.functional as F
from tqdm import tqdm

class DDPMPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet,
                              scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t).simulate_data

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).new_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

from graph_bridges.models.pipelines.pipelines_utils import register_pipeline
from graph_bridges.models.schedulers.scheduling_sb import SBScheduler
from graph_bridges.data.dataloaders import BridgeDataLoader, SpinsDataLoader
from graph_bridges.models.reference_process.ctdd_reference import ReferenceProcess
from graph_bridges.models.backward_rates.backward_rate import BackwardRate


@register_pipeline
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
    data: SpinsDataLoader
    target: BridgeDataLoader
    scheduler: SBScheduler

    def __init__(self,
                 config:SBConfig,
                 reference_process:ReferenceProcess,
                 data:SpinsDataLoader,
                 target:BridgeDataLoader,
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

    def paths_iterator(self,
                       past_model: Union[BackwardRate,ReferenceProcess] = None,
                       sinkhorn_iteration = 0,
                       device :torch.device = None,
                       train: bool = True,
                       return_dict: bool = True,
                       return_path:bool = True,
                       return_path_shape:bool = False):

        if past_model is None:
            assert sinkhorn_iteration == 0
        else:
            if isinstance(past_model,BackwardRate):
                assert sinkhorn_iteration >= 1

        # set step values
        self.scheduler.set_timesteps(self.bridge_config.sampler.num_steps,
                                     self.bridge_config.sampler.min_t,
                                     sinkhorn_iteration=sinkhorn_iteration)
        timesteps_ = self.scheduler.timesteps
        data_iterator = self.select_data_iterator(sinkhorn_iteration, train)

        for x in data_iterator:
            x = x[0]
            num_of_paths = x.shape[0]

            spins = (-1.)**(x.squeeze().float()+1)

            if return_path:
                full_path = [spins.unsqueeze(1)]

            for idx, t in tqdm(enumerate(timesteps_[0:-1])):

                h = self.select_time_difference(sinkhorn_iteration,timesteps_,idx)
                times_ = t * torch.ones(num_of_paths)

                if sinkhorn_iteration != 0:
                    logits = past_model.stein_binary_forward(spins,times_)
                    rates_ = F.softplus(logits)
                else:
                    rates_ = self.reference_process.rates_states_and_times(spins,times_)

                spins_new = self.scheduler.step(rates_,spins,t,h,device,return_dict=True,step_type="Poisson").new_sample
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
        past_model: Optional[BackwardRate] = None,
        sinkhorn_iteration = 0,
        device :torch.device = None,
        initial_spins: torch.Tensor = None,
        train: bool =True,
        return_dict: bool = True,
        return_path:bool = True,
        return_path_shape:bool = False,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """

        :param past_model:
        :param sinkhorn_iteration:
        :param device:
        :param x:
        :param train:
        :param return_dict:
        :param return_path: bool  gives you the whole path, otherwise is set to zero
        :param return_path_shape: returns paths as [batchsize,num_timesteps,dimension]
        :return:
        """
        if past_model is None:
            assert sinkhorn_iteration == 0
        else:
            if isinstance(past_model,BackwardRate):
                assert sinkhorn_iteration >= 1

        # set step values
        self.scheduler.set_timesteps(self.bridge_config.sampler.num_steps,
                                     self.bridge_config.sampler.min_t,
                                     sinkhorn_iteration=sinkhorn_iteration)
        timesteps = self.scheduler.timesteps

        if initial_spins is None:
            data_iterator = self.select_data_iterator(sinkhorn_iteration,train)
            initial_spins = next(data_iterator)[0]

        num_of_paths = initial_spins.shape[0]
        if return_path:
            full_path = [initial_spins.unsqueeze(1)]

        for idx, t in tqdm(enumerate(timesteps[0:-1])):

            h = self.select_time_difference(sinkhorn_iteration, timesteps, idx)
            times = t * torch.ones(num_of_paths)

            if sinkhorn_iteration != 0:
                logits = past_model.stein_binary_forward(initial_spins, times)
                rates_ = F.softplus(logits)
            else:
                rates_ = self.reference_process.rates_states_and_times(initial_spins, times)

            spins_new = self.scheduler.step(rates_, initial_spins, t, h, device, return_dict=True, step_type=self.bridge_config.sampler.step_type).new_sample
            initial_spins = spins_new

            if return_path:
                full_path.append(initial_spins.unsqueeze(1))

        if return_path:
            full_path = torch.concat(full_path, dim=1)#
            number_of_paths = full_path.shape[0]
            number_of_timesteps = full_path.shape[1]

            timesteps = timesteps.unsqueeze(0).repeat(num_of_paths, 1)
            if return_path_shape:
                return full_path, timesteps
            else:
                timesteps = timesteps.reshape(number_of_paths * number_of_timesteps)
                full_path = full_path.reshape(number_of_paths * number_of_timesteps, -1)
                return full_path, timesteps

        else:
            return spins_new