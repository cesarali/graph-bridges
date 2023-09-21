import torch
from torch import nn
from dataclasses import dataclass
from typing import Union


from graph_bridges.models.temporal_networks.embedding_utils import transformer_timestep_embedding

@dataclass
class TemporalMLPConfig:
    temp_name:str = "TemporalMLP"

    time_embed_dim : int = 9
    hidden_dim : int = 200

    do_time_embed: bool = True
    time_embed_dim: int = 128
    fix_logistic :bool = False


class TemporalMLP(nn.Module):

    def __init__(self,config,device):
        super().__init__()

        self.time_embed_dim = config.temp_network.time_embed_dim
        self.hidden_layer = config.temp_network.hidden_dim
        self.num_states = config.data.number_of_states
        self.dimension = config.data.D
        self.define_deep_models()

        self.device = device

    def define_deep_models(self):
        # layers
        self.f1 = nn.Linear(self.dimension, self.hidden_layer)
        self.f2 = nn.Linear(self.hidden_layer + self.time_embed_dim, self.dimension * self.num_states)

    def forward(self,x,times):
        batch_size = x.shape[0]
        time_embbedings = transformer_timestep_embedding(times,
                                                         embedding_dim=self.time_embed_dim)

        step_one = self.f1(x)
        step_two = torch.concat([step_one, time_embbedings], dim=1)
        rate_logits = self.f2(step_two)
        rate_logits = rate_logits.reshape(batch_size,self.dimension,self.num_states)

        return rate_logits