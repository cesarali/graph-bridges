import os
import torch
import pandas as pd
from pathlib import Path
from pprint import pprint
from dataclasses import dataclass, asdict

from graph_bridges.configs.graphs.config_sb import BridgeConfig
from graph_bridges.data.dataloaders_config import GraphSpinsDataLoaderConfig
from graph_bridges.models.reference_process.ctdd_reference import GaussianTargetRate
from graph_bridges.models.reference_process.reference_process_config import GaussianTargetRateConfig
from graph_bridges.data.dataloaders_utils import create_dataloader
from graph_bridges.data.dataloaders import GraphSpinsDataLoader

if __name__=="__main__":
    bridge_config = BridgeConfig()
    bridge_config.reference = GaussianTargetRateConfig()
    bridge_config.data = GraphSpinsDataLoaderConfig()

    device = torch.device("cpu")
    data_loader = GraphSpinsDataLoader(bridge_config.data, device,0)
    x_spins = next(data_loader.train().__iter__())[0]

    print(x_spins.shape)
    reference_process = GaussianTargetRate(bridge_config,device)

