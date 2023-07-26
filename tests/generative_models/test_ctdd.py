import os
import torch
import numpy as np
import pandas as pd

from pprint import pprint
from dataclasses import asdict

if __name__=="__main__":
    from graph_bridges.configs.graphs.config_ctdd import CTDDConfig
    from graph_bridges.models.generative_models.ctdd import CTDD
    from graph_bridges.data.graph_dataloaders_config import EgoConfig, GraphSpinsDataLoaderConfig, TargetConfig
    from graph_bridges.models.backward_rates.backward_rate_config import BackRateMLPConfig, GaussianTargetRateImageX0PredEMAConfig

    device = torch.device("cpu")
    config = CTDDConfig()
    config.data = EgoConfig()
    config.model = GaussianTargetRateImageX0PredEMAConfig()

    ctdd = CTDD()
    ctdd.create_from_config(config,device)


