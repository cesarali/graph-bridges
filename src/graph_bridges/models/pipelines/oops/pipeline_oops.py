import torch
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional,Union,Tuple
from graph_bridges.configs.config_oops import OopsConfig
from graph_bridges.models.networks_arquitectures.rbf import RBM

from graph_bridges.data.image_dataloader_config import NISTLoaderConfig
from graph_bridges.data.image_dataloaders import NISTLoader
import torch.distributions as dists

from graph_bridges.utils.test_utils import check_model_devices


class OopsPipeline:
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """
    config : OopsConfig
    model: RBM
    data: NISTLoader

    def __init__(self,
                 config:OopsConfig,
                 model:RBM,
                 data:NISTLoaderConfig,
                 data_mean=None,
                 device=None):

        super().__init__()
        self.oops_config= config

        self.D = self.oops_config.data.D
        self.rbf = model
        self.data = data

        if device is None:
            self.device = check_model_devices(model)
        else:
            self.device = device

        if data_mean is not None:
            init_val = (data_mean / (1. - data_mean)).log()
            self.b_v.data = init_val
            self.init_dist = dists.Bernoulli(probs=data_mean)
        else:
            self.init_dist = dists.Bernoulli(probs=torch.ones((self.D,)) * .5)

    @torch.no_grad()
    def __call__(
        self,
        model: Optional[RBM] = None,
        sample_size = None,
        return_dict: bool = True,
        device :torch.device = None,
    ) -> Union[torch.Tensor, Tuple]:
        """

        :param model:
        :param sinkhorn_iteration:
        :param return_dict:
        :param device:
        :return:
        """
        if device is None:
            device = self.device
        if sample_size is None:
            sample_size = self.oops_config.pipeline.num_samples

        x = self.gibbs_sample(num_gibbs_steps=self.oops_config.pipeline.num_gibbs_steps,
                              num_samples=sample_size,
                              device=device)

        return x

    def _gibbs_step(self, v):
        h = self.rbf.p_h_given_v(v).sample()
        v = self.rbf.p_v_given_h(h).sample()
        return v

    def gibbs_sample(self, v=None, num_gibbs_steps=2000, num_samples=None,device=torch.device("cpu")):
        if v is None:
            assert num_samples is not None
            v = self.init_dist.sample((num_samples,)).to(device)
        for i in range(num_gibbs_steps):
            v = self._gibbs_step(v)
        return v