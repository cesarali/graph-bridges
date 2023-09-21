import os
import torch
from torch import nn
from dataclasses import dataclass
from typing import Union

#from graph_bridges.configs.config_sb import SBConfig
#from graph_bridges.configs.config_ctdd import CTDDConfig

from graph_bridges.models.temporal_networks.embedding_utils import transformer_timestep_embedding
from graph_bridges.models.temporal_networks.transformers.hollow_transformers import HollowTransformerLayer

@dataclass
class TemporalHollowTransformerConfig:
    temp_name:str = "TemporalHollowTransformer"
    num_layers: int = 4
    num_heads: int = 8
    hidden_dim: int = 256
    ff_hidden_dim: int = 512
    input_vocab_size: int = 10000
    output_vocab_size: int = 10000
    max_seq_length: int = 50
    do_time_embed: bool = True
    time_embed_dim: int = 128
    time_scale_factor: float = 1000
    fix_logistic :bool = False


class TemporalHollowTransformer(nn.Module):

    def __init__(self, config, device:torch.device):
        super(TemporalHollowTransformer, self).__init__()

        self.device = device

        self.num_layers = config.temp_network.num_layers
        self.num_heads = config.temp_network.num_heads
        self.hidden_dim = config.temp_network.hidden_dim
        self.ff_hidden_dim = config.temp_network.ff_hidden_dim
        self.input_vocab_size = config.temp_network.input_vocab_size
        self.max_seq_length = config.temp_network.max_seq_length
        self.output_vocab_size = config.temp_network.output_vocab_size
        self.time_embed_dim = config.temp_network.time_embed_dim
        self.time_scale_factor = config.temp_network.time_scale_factor
        self.do_time_embed = config.temp_network.do_time_embed
        self.define_time_embeddings()

        self.embedding = nn.Embedding(self.input_vocab_size, self.hidden_dim).to(self.device)
        self.transformer_layers = nn.ModuleList(
            [HollowTransformerLayer(self.num_heads,
                                    self.hidden_dim,
                                    self.ff_hidden_dim,
                                    self.expanded_time_dim).to(self.device) for _ in range(self.num_layers)]
        )
        self.fc = nn.Linear(self.hidden_dim, self.output_vocab_size).to(self.device)
        self.to(self.device)
    def forward(self, x, timesteps, mask=None):
        temb = self._time_embedding(timesteps)
        x = self.embedding(x)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x,mask,temb)
        x = self.fc(x)
        return x

    def define_time_embeddings(self):
        self.temb_modules = []
        self.temb_modules.append(nn.Linear(self.time_embed_dim, self.time_embed_dim*4))
        nn.init.zeros_(self.temb_modules[-1].bias)
        self.temb_modules.append(nn.Linear(self.time_embed_dim*4, self.time_embed_dim*4))
        nn.init.zeros_(self.temb_modules[-1].bias)
        self.temb_modules = nn.ModuleList(self.temb_modules).to(self.device)
        self.act = nn.functional.silu

        self.expanded_time_dim = 4 * self.time_embed_dim if self.do_time_embed else None

    def _time_embedding(self, timesteps):
        if self.do_time_embed:
            temb = transformer_timestep_embedding(
                timesteps * self.time_scale_factor, self.time_embed_dim
            )
            temb = self.temb_modules[0](temb)
            temb = self.temb_modules[1](self.act(temb))
        else:
            temb = None

        return temb