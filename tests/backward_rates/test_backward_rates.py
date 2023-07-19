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

    forward_ = model(x_spins.squeeze(),times)
    print(forward_.shape)

    #config.model = BackRateMLPConfig()
    #model = all_backward_rates[config.model.name](config,device)

    from graph_bridges.data.graph_dataloaders_config import CommunityConfig
    from dataclasses import asdict

    data_config = CommunityConfig()
    print(asdict(data_config))
    train_loader, test_loader = load_data(data_config)

    databatch = next(train_loader.__iter__())
    features_ = databatch[0]
    adjacencies_ = databatch[1]

    print(features_.shape)
    print(adjacencies_.shape)

    """

    """