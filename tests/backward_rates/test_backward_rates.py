import os
import torch
from pathlib import Path
from pprint import pprint
from dataclasses import dataclass, asdict

from graph_bridges.data.graph_dataloaders import load_data



if __name__=="__main__":

    from graph_bridges.models.backward_rates.backward_rate_config import BackRateMLPConfig,GaussianTargetRateImageX0PredEMAConfig
    from graph_bridges.models.backward_rates.backward_rate import all_backward_rates
    from graph_bridges.data.dataloaders_config import GraphSpinsDataLoaderConfig
    from graph_bridges.configs.graphs.config_sb import BridgeConfig
    from graph_bridges.data.dataloaders_utils import create_dataloader
    from graph_bridges.data.dataloaders import all_dataloaders

    config = BridgeConfig(experiment_indentifier="debug")
    config.data = GraphSpinsDataLoaderConfig()
    device = torch.device("cpu")

    print(asdict(config.data))

    data_loader = all_dataloaders[config.data.name](config.data,device,0)
    x_spins = next(data_loader.train().__iter__())[0]
    batch_size = x_spins.shape[0]
    times = torch.rand((batch_size))

    #forward_ = model(x_spins_data.squeeze(),times)
    #forward_stein = model.stein_binary_forward(x_spins_data.squeeze(),times)
    config.model = GaussianTargetRateImageX0PredEMAConfig()
    model = all_backward_rates[config.model.name](config,device)

    x_spins_ = x_spins.squeeze()
    forward_ = model(x_spins_,times)
    print("Adjacency")
    print(x_spins_.shape)
    print("Times")
    print(times.shape)
    print("Forward")
    print(forward_.shape)

    #config.model = BackRateMLPConfig()
    #model = all_backward_rates[config.model.name](config,device)

    #===================================================
    # DATALOADERS
    #===================================================

    from graph_bridges.data.graph_dataloaders_config import CommunityConfig
    from graph_bridges.data.graph_dataloaders import BridgeGraphDataLoaders

    bridge_config = BridgeConfig(experiment_indentifier="debug")
    bridge_config.data = CommunityConfig()
    bridge_config.model = GaussianTargetRateImageX0PredEMAConfig()

    bridge_graph_dataloader = BridgeGraphDataLoaders(bridge_config,device)

    databatch = next(bridge_graph_dataloader.train().__iter__())
    adj = databatch[0]
    features = databatch[1]


    """

    """