import os
import torch
from pathlib import Path
from pprint import pprint
from dataclasses import dataclass, asdict
from graph_bridges.models.backward_rates.backward_rate import BackwardRate
from graph_bridges.data.dataloaders_utils import load_dataloader



if __name__=="__main__":
    from graph_bridges.models.generative_models.sb import SB
    from graph_bridges.data.graph_dataloaders_config import EgoConfig
    from graph_bridges.models.backward_rates.backward_rate_utils import load_backward_rates
    from graph_bridges.models.backward_rates.backward_rate_config import BackRateMLPConfig
    from graph_bridges.configs.graphs.config_sb import SBConfig, ParametrizedSamplerConfig, SteinSpinEstimatorConfig
    from graph_bridges.configs.graphs.config_sb import TrainerConfig
    from graph_bridges.models.backward_rates.backward_rate_config import GaussianTargetRateImageX0PredEMAConfig

    device = torch.device("cpu")
    config = SBConfig(experiment_indentifier="debug")
    config.data = EgoConfig(as_image=False, batch_size=5, full_adjacency=False,as_spins=False)
    config.model = BackRateMLPConfig(time_embed_dim=12,hidden_layer=300)
    config.stein = SteinSpinEstimatorConfig(stein_sample_size=20)
    config.sampler = ParametrizedSamplerConfig(num_steps=10)
    config.optimizer = TrainerConfig(learning_rate=1e-3,num_epochs=400)
    config.align_configurations()

    # ===================================================
    # DATALOADERS
    # ===================================================
    data_dataloader = load_dataloader(config,type="data",device=device)
    model = load_backward_rates(config,device)
    databatch = next(data_dataloader.train().__iter__())

    x_adj = databatch[0]
    forward_ = model(x_adj,data_dataloader.fake_time_)
    forward_stein = model.stein_binary_forward(x_adj,data_dataloader.fake_time_)

    print(forward_stein.mean())
    #=====================================================
    # OTHER MODEL
    #=====================================================

    config = SBConfig(experiment_indentifier="debug")
    config.data = EgoConfig(as_image=False, batch_size=5, full_adjacency=False, as_spins=False)
    config.model = GaussianTargetRateImageX0PredEMAConfig(time_embed_dim=12,)
    config.stein = SteinSpinEstimatorConfig(stein_sample_size=20)
    config.sampler = ParametrizedSamplerConfig(num_steps=10)
    config.align_configurations()

    # ===================================================
    # DATALOADERS
    # ===================================================
    data_dataloader = load_dataloader(config, type="data", device=device)
    model = load_backward_rates(config, device)
    databatch = next(data_dataloader.train().__iter__())

    x_adj = databatch[0]
    forward_ = model(x_adj, data_dataloader.fake_time_)
    forward_stein = model.stein_binary_forward(x_adj, data_dataloader.fake_time_)
    print(forward_stein.mean())


