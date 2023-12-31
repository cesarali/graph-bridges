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

from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from graph_bridges.configs.config_ctdd import CTDDConfig
from diffusers.utils import randn_tensor
import torch.nn.functional as F
from tqdm import tqdm

from typing import List, Optional, Tuple, Union
import torch


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
        if isinstance(self.unet.sb_config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.sb_config.in_channels,
                self.unet.sb_config.sample_size,
                self.unet.sb_config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.sb_config.in_channels, *self.unet.sb_config.sample_size)

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

from graph_bridges.models.schedulers.scheduling_ctdd import CTDDScheduler
from graph_bridges.data.graph_dataloaders import DoucetTargetData,BridgeGraphDataLoaders
from graph_bridges.models.reference_process.ctdd_reference import ReferenceProcess
from graph_bridges.models.backward_rates.ctdd_backward_rate import BackwardRate


class CTDDPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """
    config : CTDDConfig
    model: BackwardRate
    reference_process: ReferenceProcess
    data: BridgeGraphDataLoaders
    target: DoucetTargetData
    scheduler: CTDDScheduler

    def __init__(self,
                 config:CTDDConfig,
                 reference_process:ReferenceProcess,
                 data:BridgeGraphDataLoaders,
                 target:DoucetTargetData,
                 scheduler:CTDDScheduler):

        super().__init__()
        self.register_modules(reference_process=reference_process,
                              data=data,
                              target=target,
                              scheduler=scheduler)
        self.ctdd_config = config
        self.D = self.ctdd_config.data.D

    @torch.no_grad()
    def __call__(
        self,
        model: Optional[BackwardRate] = None,
        sample_size = None,
        return_dict: bool = True,
        device :torch.device = None,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """

        :param model:
        :param sinkhorn_iteration:
        :param return_dict:
        :param device:
        :return:
        """
        # Sample gaussian noise to begin loop
        if sample_size is None:
            num_of_paths = self.ctdd_config.number_of_paths
        else:
            num_of_paths = sample_size
        x = self.target.sample(num_of_paths, device)[0].float()

        # set step values
        self.scheduler.set_timesteps(self.ctdd_config.sampler.num_steps,
                                     self.ctdd_config.sampler.min_t)
        timesteps = self.scheduler.timesteps
        timesteps = timesteps.to(x.device)

        for idx, t in tqdm(enumerate(timesteps[0:-1])):
            h = timesteps[idx] - timesteps[idx + 1]
            # h = h.to(x.device)
            # t = t.to(x.device)
            t = t*torch.ones((num_of_paths,), device=x.device)
            logits = model(x, t)
            p0t = F.softmax(logits,dim=2)  # (N, D, S)
            rates_ = self.reference_process.backward_rates_from_probability(p0t, x, t, device)

            x_new = self.scheduler.step(rates_,x,t,h).prev_sample
            x = x_new

        return x