import torch
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from abc import ABC,abstractmethod
from graph_bridges.configs.graphs.graph_config_sb import SBConfig

from typing import Union, Tuple
from torchtyping import TensorType
from graph_bridges.models.temporal_networks import networks_tau
from graph_bridges.models.reference_process.ctdd_reference import GaussianTargetRate


from dataclasses import dataclass
from diffusers.utils import BaseOutput
from graph_bridges.configs.config_sb import SBConfig

class SBBackwardRate(nn.Module,ABC):

    def __init__(self,
                 config:SBConfig,
                 device,
                 rank,
                 **kwargs):
        super().__init__()

        self.config = config

        # DATA
        self.dimension = config.data.D
        self.num_states = config.data.S
        #self.data_type = config.data.type
        self.data_min_max = config.data.data_min_max

        # TIME
        self.time_embed_dim = config.temp_network.time_embed_dim
        self.act = nn.functional.silu

    def init_parameters(self):
        return None

    @abstractmethod
    def _forward(self,
                x: TensorType["batch_size", "dimension"],
                times:TensorType["batch_size"]
                )-> TensorType["batch_size", "dimension", "num_states"]:
        return None

    def _center_data(self, x):
        out = (x - self.data_min_max[0]) / (self.data_min_max[1] - self.data_min_max[0])  # [0, 1]
        return 2 * out - 1  # to put it in [-1, 1]

    def forward(self,
                x: TensorType["batch_size", "dimension"],
                times:TensorType["batch_size"],
                x_tilde: TensorType["batch_size", "dimension"] = None,
                return_dict: bool = False,
                )-> Union[BackwardRateOutput, torch.FloatTensor, Tuple]:
        if x_tilde is not None:
            return self.ctdd(x,x_tilde,times,return_dict)
        else:
            x_logits = self._forward(x, times)
            if not return_dict:
                return x_logits
            else:
                return BackwardRateOutput(x_logits=x_logits, p0t_reg=None, p0t_sig=None)

    def ctdd(self,x_t,x_tilde,times,return_dict):
        if x_tilde is not None:
            if self.config.loss.one_forward_pass:
                reg_x = x_tilde
                x_logits = self._forward(reg_x, times)  # (B, D, S)
                p0t_reg = F.softmax(x_logits, dim=2)  # (B, D, S)
                p0t_sig = p0t_reg
            else:
                reg_x = x_t
                x_logits = self._forward(reg_x, times)  # (B, D, S)
                p0t_reg = F.softmax(x_logits, dim=2)  # (B, D, S)
                p0t_sig = F.softmax(self._forward(x_tilde, times), dim=2)  # (B, D, S)

            if not return_dict:
                return (x_logits,p0t_reg,p0t_sig,reg_x)
            return BackwardRateOutput(x_logits=x_logits,p0t_reg=p0t_reg,p0t_sig=p0t_sig,reg_x=reg_x)
        else:
            return self._forward(x_t, times)  # (B, D, S)

    def stein_binary_forward(self,
                x: TensorType["batch_size", "dimension"],
                times: TensorType["batch_size"]
                ) -> TensorType["batch_size", "dimension"]:
        forward_logits = self.forward(x,times)
        return forward_logits[:,:,0]