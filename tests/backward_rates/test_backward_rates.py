import os
import torch
from pathlib import Path
from pprint import pprint
from dataclasses import dataclass, asdict
from graph_bridges.models.backward_rates.backward_rate import BackwardRate
from graph_bridges.data.dataloaders_utils import load_dataloader



if __name__=="__main__":
    from graph_bridges.models.backward_rates.backward_rate_config import BackRateMLPConfig,GaussianTargetRateImageX0PredEMAConfig
    from graph_bridges.models.backward_rates.backward_rate import all_backward_rates
    from graph_bridges.data.dataloaders_config import GraphSpinsDataLoaderConfig
    from graph_bridges.configs.graphs.config_sb import SBConfig
    from graph_bridges.data.dataloaders_utils import create_dataloader
    from graph_bridges.data.dataloaders import all_dataloaders

    # ===================================================
    # DATALOADERS
    # ===================================================

    from graph_bridges.data.graph_dataloaders_config import CommunityConfig, EgoConfig
    from graph_bridges.data.graph_dataloaders import BridgeGraphDataLoaders

    device = torch.device("cpu")

    bridge_config = SBConfig(experiment_indentifier="debug")
    graph_config = EgoConfig(full_adjacency=True, flatten_adjacency=True, as_image=False, as_spins=False)

    bridge_config.data = graph_config
    bridge_graph_dataloader = BridgeGraphDataLoaders(bridge_config, device)
    databatch = next(bridge_graph_dataloader.train().__iter__())
    adj_states, features = databatch[0], databatch[1]

    batch_size = adj_states.shape[0]
    times = torch.rand((batch_size))

    bridge_config.model = GaussianTargetRateImageX0PredEMAConfig()
    model: BackwardRate
    model = all_backward_rates[bridge_config.model.name](bridge_config,device)

    forward_ = model(adj_states,times)
    forward_stein = model.stein_binary_forward(adj_states,times)

    print("Adjacency")
    print(adj_states.shape)
    print("Times")
    print(times.shape)
    print("Forward")
    print(forward_.shape)
    print("Forward Stein")
    print(forward_stein.shape)

    #config.model = BackRateMLPConfig()
    #model = all_backward_rates[config.model.name](config,device)

    bridge_config.model = GaussianTargetRateImageX0PredEMAConfig()
    databatch = next(bridge_graph_dataloader.train().__iter__())
    adj = databatch[0]
    features = databatch[1]


