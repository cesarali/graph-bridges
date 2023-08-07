import os
import json
import torch
from graph_bridges.models.generative_models.sb import SB
from graph_bridges.models.backward_rates.backward_rate import GaussianTargetRateImageX0PredEMA
from graph_bridges.models.backward_rates.backward_rate_config import GaussianTargetRateImageX0PredEMAConfig
import networkx as nx

if __name__=="__main__":
    from graph_bridges.configs.graphs.config_sb import SBConfig
    from graph_bridges.data.graph_dataloaders_config import EgoConfig
    from graph_bridges.models.backward_rates.backward_rate_config import BackRateMLPConfig
    from graph_bridges.configs.graphs.config_sb import ParametrizedSamplerConfig

    config = SBConfig(experiment_indentifier="debug")
    config.data = EgoConfig(as_image=False, batch_size=32, full_adjacency=False)
    config.model = GaussianTargetRateImageX0PredEMAConfig(time_embed_dim=32, fix_logistic=False)
    config.sampler = ParametrizedSamplerConfig(num_steps=18)


    #read the model
    #config = get_config_from_file("graph", "lobster", "1687884918")
    device = torch.device(config.device)
    sb = SB(config,device)

    from graph_bridges.models.metrics.sb_metrics import graph_metrics_for_sb
    metrics_ = graph_metrics_for_sb(sb,sb.training_model,device)
    print(metrics_)
