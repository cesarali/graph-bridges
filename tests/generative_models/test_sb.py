import os
import sys
import torch

if __name__=="__main__":
    from graph_bridges.data.dataloaders_utils import load_dataloader
    from graph_bridges.models.backward_rates.backward_rate_utils import load_backward_rates

    from graph_bridges.models.generative_models.sb import SB
    from graph_bridges.configs.graphs.config_sb import SBConfig

    from graph_bridges.data.graph_dataloaders_config import EgoConfig, GraphSpinsDataLoaderConfig, TargetConfig
    from graph_bridges.models.backward_rates.backward_rate_config import BackRateMLPConfig, GaussianTargetRateImageX0PredEMAConfig

    device = torch.device("cpu")

    config = SBConfig(experiment_indentifier="testing")
    config.data = EgoConfig(as_image=False, batch_size=32, full_adjacency=False)
    config.model = GaussianTargetRateImageX0PredEMAConfig(time_embed_dim=32, fix_logistic=False)
    config.initialize_new_experiment()

    sb = SB()
    sb.create_new_from_config(config, device)
    x = sb.pipeline(sb.model,sample_size=36)
