import torch
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional,Union,Tuple
from graph_bridges.models.networks_arquitectures.rbf import RBM
from graph_bridges.configs.config_oops import OopsConfig

from graph_bridges.data.image_dataloader_config import NISTLoaderConfig
from graph_bridges.data.image_dataloaders import NISTLoader

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
                 data:NISTLoaderConfig):

        super().__init__()
        self.oops_config= config

        self.D = self.oops_config.data.D
        self.rbf = model
        self.data = data

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
        x = self.gibbs_sample(num_gibbs_steps=self.n)
        return x

    def _gibbs_step(self, v):
        h = self.rbf.p_h_given_v(v).sample()
        v = self.rbf.p_v_given_h(h).sample()
        return v

    def gibbs_sample(self, v=None, num_gibbs_steps=2000, num_samples=None, plot=False):
        if v is None:
            assert num_samples is not None
            v = self.init_dist.sample((num_samples,)).to(self.W.device)
        for i in range(num_gibbs_steps):
            v = self._gibbs_step(v)
        return v