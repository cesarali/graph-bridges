import os
import sys
import torch

if __name__=="__main__":
    from graph_bridges.data.dataloaders_utils import load_dataloader
    from graph_bridges.models.backward_rates.backward_rate_utils import load_backward_rates

    from graph_bridges.models.generative_models.sb import SB
    from graph_bridges.data.graph_dataloaders_config import EgoConfig
    from graph_bridges.models.backward_rates.backward_rate_config import BackRateMLPConfig
    from graph_bridges.configs.graphs.config_sb import SBConfig, ParametrizedSamplerConfig, SteinSpinEstimatorConfig
    from graph_bridges.models.backward_rates.backward_rate_config import GaussianTargetRateImageX0PredEMAConfig

    config = SBConfig(experiment_indentifier="debug")
    config.data = EgoConfig(as_image=False, batch_size=5, full_adjacency=False)
    config.model = GaussianTargetRateImageX0PredEMAConfig(time_embed_dim=12, fix_logistic=False)
    config.stein = SteinSpinEstimatorConfig(stein_sample_size=20)
    config.sampler = ParametrizedSamplerConfig(num_steps=10)

    #read the model
    device = torch.device("cpu")
    sb = SB(config, device)
    spins_path_1, times_1 = sb.pipeline(sb.training_model,
                                        sinkhorn_iteration=1,
                                        device=device,
                                        train=True,
                                        return_path=False)
    #generated_graphs = sb.generate_graphs(100)
    #print(len(generated_graphs))