import torch
import torch.nn as nn
import numpy as np
from graph_bridges.models.networks.graphs_networks.edp_gnn import EdgeDensePredictionGraphScoreNetwork

from graph_bridges.models.backward_rates.backward_rate import EMA, BackwardRate, GaussianTargetRate

class BackRateMLP(EMA,BackwardRate,GaussianTargetRate):

    def __init__(self, config, device, rank=None):
        
        EMA.__init__(self, config)
        BackwardRate.__init__(self, config, device, rank)

        self.hidden_layer = config.model.hidden_layer
        self.define_deep_models()
        self.init_ema()
        self.to(device)

    def define_deep_models(self):
        self.f1 = nn.Linear(self.dimension, self.hidden_layer)
        self.f2 = nn.Linear(self.hidden_layer + self.time_embed_dim,self.dimension*self.num_states)

    def _forward(self,
                x: TensorType["batch_size", "dimension"],
                times: TensorType["batch_size"]
                ) -> TensorType["batch_size", "dimension", "num_states"]:

        if self.config.data.type == "doucet":
            x = self._center_data(x)

        batch_size = x.shape[0]
        time_embbedings = transformer_timestep_embedding(times,
                                                         embedding_dim=self.time_embed_dim)

        step_one = self.f1(x)
        step_two = torch.concat([step_one, time_embbedings], dim=1)
        rate_logits = self.f2(step_two)
        rate_logits = rate_logits.reshape(batch_size,self.dimension,self.num_states)
        return rate_logits

    def init_parameters(self):
        nn.init.xavier_uniform_(self.f1.weight)
        nn.init.xavier_uniform_(self.f2.weight)
