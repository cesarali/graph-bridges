import torch
import math
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
from graph_bridges.models.temporal_networks.temporal_networks_utils import load_temp_network
from graph_bridges.models.temporal_networks.transformers.temporal_hollow_transformers import TemporalHollowTransformerConfig
from graph_bridges.models.temporal_networks.ema import EMA


from dataclasses import dataclass
from diffusers.utils import BaseOutput
from graph_bridges.configs.config_sb import SBConfig
from graph_bridges.data.transforms import SpinsToBinaryTensor

spins_to_binary_tensor = SpinsToBinaryTensor()

class SchrodingerBridgeBackwardRate(EMA,nn.Module):
    """
    Schrödinger Bridge Rates Defines Flip Only Rates after temporal network logits
    """
    def __init__(self,
                 config:SBConfig,
                 device:torch.device):
        EMA.__init__(self, config)
        nn.Module.__init__(self)

        self.config = config
        self.init_ema()

        # DATA
        self.temporal_network_shape = torch.Size(config.data.shape_)
        self.dimension = config.data.D
        self.num_states = config.data.S

        if self.num_states != 2:
            raise Exception("Schrodinger Bridge Implemented for Spins Only")

        self.data_min_max = config.data.data_min_max
        self.device = device

        # TIME
        self.time_embed_dim = config.temp_network.time_embed_dim
        self.temp_network = load_temp_network(self.config,self.device)
        if isinstance(self.temp_network.expected_output_shape,list):
            self.expected_output_dim = math.prod(self.temp_network.expected_output_shape)
        self.flip_rate_logits = nn.Linear(self.expected_output_dim,self.dimension).to(self.device)


    def forward(self,
                x: TensorType["batch_size", "dimension"],
                times:TensorType["batch_size"],
                )-> torch.FloatTensor:
        batch_size = x.shape[0]

        if x.shape[1:] != self.temporal_network_shape:
            data_size = list(self.temporal_network_shape)
            data_size.insert(0, batch_size)
            data_size = torch.Size(data_size)
            x = x.reshape(data_size)

        if isinstance(self.config.temp_network,TemporalHollowTransformerConfig):
            x = spins_to_binary_tensor(x).long()

        temporal_net_logits = self.temp_network(x, times)
        flip_rate_logits = self.flip_rate_logits(temporal_net_logits.view(batch_size,-1))
        flip_rates = F.softplus(flip_rate_logits)
        return flip_rates

    def stein_binary_forward(self,
                x: TensorType["batch_size", "dimension"],
                times: TensorType["batch_size"]
                ) -> TensorType["batch_size", "dimension"]:
        flip_rate_logits = self.forward(x,times)
        return flip_rate_logits