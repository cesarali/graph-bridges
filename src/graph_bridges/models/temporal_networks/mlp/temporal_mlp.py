import torch
from torch import nn
from dataclasses import dataclass,fields,field
from typing import Union
from typing import List

from graph_bridges.models.backward_rates.sb_backward_rate_config import SchrodingerBridgeBackwardRateConfig
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
        self.num_states = config.data.S
        self.dimension = config.data.D
        self.expected_output_shape = [self.dimension,self.num_states]

        if isinstance(config.model,SchrodingerBridgeBackwardRateConfig):
            self.expected_output_shape = [self.dimension, 1]
            self.num_states = 1

        self.define_deep_models()
        self.device = device
        self.to(self.device)
        self.init_weights()

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

    def init_weights(self):
        nn.init.xavier_uniform_(self.f1.weight)
        nn.init.xavier_uniform_(self.f2.weight)

@dataclass
class DeepTemporalMLPConfig:
    temp_name:str = "DeepTemporalMLP"

    layers_dim : List[int] = field(default_factory=lambda:[100,100])
    time_embed_hidden: int = 200
    data_hidden:int = 200
    output_transformation:str=None # softplus, sigmoid, relu
    dropout: float = 0.2
    do_time_embed: bool = True
    time_embed_dim: int = 128
    fix_logistic :bool = False

class DeepTemporalMLP(nn.Module):
    def __init__(self,config,device:torch.device):
        nn.Module.__init__(self)

        self.time_embed_dim = config.temp_network.time_embed_dim
        self.layers_dim = config.temp_network.layers_dim
        self.dropout = config.temp_network.dropout
        self.time_embed_hidden = config.temp_network.time_embed_hidden
        self.data_hidden = config.temp_network.data_hidden
        self.num_states = config.data.S
        self.dimension = config.data.D
        self.expected_output_shape = [self.dimension,self.num_states]
        self.device = device
        if isinstance(config.model,SchrodingerBridgeBackwardRateConfig):
            self.expected_output_shape = [self.dimension, 1]
            self.num_states = 1

        self.input_dim = self.time_embed_hidden + self.data_hidden
        self.output_dim = self.dimension * self.num_states
        #self.normalization = config.temp_network.normalization

        self.layers_dim = [self.input_dim] + self.layers_dim + [self.output_dim]
        self.num_layer = len(self.layers_dim)
        self.output_transformation = config.temp_network.output_transformation
        self.define_deep_parameters()

        self.to(self.device)
    def forward(self,x,times):
        time_embbedings = transformer_timestep_embedding(times,
                                                         embedding_dim=self.time_embed_dim)
        time_hidden = self.encode_time_emb(time_embbedings)
        data_hidden = self.encode_data(x)
        x = torch.concat([time_hidden, data_hidden], dim=1)

        return self.perceptron(x)

    def define_deep_parameters(self):
        self.encode_data = nn.Linear(self.dimension, self.data_hidden)
        self.encode_time_emb = nn.Linear(self.time_embed_dim, self.time_embed_hidden)

        self.perceptron = nn.ModuleList([])
        for layer_index in range(self.num_layer - 1):
            self.perceptron.append(nn.Linear(self.layers_dim[layer_index], self.layers_dim[layer_index + 1]))
            if self.dropout > 0 and layer_index != self.num_layer - 2:
                self.perceptron.append(nn.Dropout(self.dropout))
            # if self.normalization and layer_index < self.num_layer - 2:
            #     self.perceptron.append(nn.BatchNorm1d(self.layers_dim[layer_index  1]))
            if layer_index != self.num_layer - 2:
                if layer_index < self.num_layer - 1 and self.num_layer > 2:
                    self.perceptron.append(nn.ReLU())
        if self.output_transformation == "relu":
            self.perceptron.append(nn.ReLU())
        elif self.output_transformation == "sigmoid":
            self.perceptron.append(nn.Sigmoid())
        elif self.output_transformation == "softplus":
            self.perceptron.append(nn.Softplus)

        self.perceptron = nn.Sequential(*self.perceptron)

    def init_parameters(self):
        nn.init.xavier_uniform_(self.encode_data.weight)
        nn.init.xavier_uniform_(self.encode_time_emb.weight)

        for layer in self.perceptron:
            if hasattr(layer, "weight"):
                if isinstance(layer, (nn.InstanceNorm2d, nn.LayerNorm)):
                    nn.init.normal_(layer.weight, mean=1.0, std=0.02)
                else:
                    nn.init.xavier_normal_(layer.weight)
            if hasattr(layer, "bias"):
                nn.init.constant_(layer.bias, 0.0)