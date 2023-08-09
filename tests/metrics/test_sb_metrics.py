import os
import json
import torch
from graph_bridges.models.generative_models.sb import SB
from graph_bridges.models.backward_rates.backward_rate import GaussianTargetRateImageX0PredEMA
from graph_bridges.models.backward_rates.backward_rate_config import GaussianTargetRateImageX0PredEMAConfig
import networkx as nx

if __name__=="__main__":
    from graph_bridges.configs.graphs.config_sb import TrainerConfig
    from graph_bridges.data.graph_dataloaders_config import EgoConfig,CommunityConfig,CommunitySmallConfig
    from graph_bridges.models.backward_rates.backward_rate_config import BackRateMLPConfig
    from graph_bridges.configs.graphs.config_sb import SBConfig, ParametrizedSamplerConfig, SteinSpinEstimatorConfig
    from graph_bridges.models.backward_rates.backward_rate_config import GaussianTargetRateImageX0PredEMAConfig

    config = SBConfig(delete=True,experiment_indentifier="testing")

    #config.data = EgoConfig(as_image=False, batch_size=32, full_adjacency=False)

    config.data = CommunityConfig(as_image=False, batch_size=32, full_adjacency=False)
    #config.data = EgoConfig(as_image=False, batch_size=32, full_adjacency=False)
    #config.data = CommunitySmallConfig(as_image=False, batch_size=32, full_adjacency=False)
    config.model = GaussianTargetRateImageX0PredEMAConfig(time_embed_dim=12, fix_logistic=False)

    #config.model = BackRateMLPConfig(time_embed_dim=14,hidden_layer=150)
    config.stein = SteinSpinEstimatorConfig(stein_sample_size=100)
    config.sampler = ParametrizedSamplerConfig(num_steps=10,step_type="TauLeaping")
    config.optimizer = TrainerConfig(learning_rate=1e-3,
                                     num_epochs=200,
                                     save_metric_epochs=20,
                                     metrics=["graphs_plots",
                                              "histograms"])
    #read the model
    device = torch.device("cpu")
    sb = SB(config, device)

    #x = sb.pipeline(sb.model,sample_size=36)
    #generated_graphs = sb.generate_graphs(100)

    from graph_bridges.models.metrics.sb_metrics import graph_metrics_and_paths_histograms,paths_marginal_histograms

    backward_histogram,forward_histogram,forward_time = paths_marginal_histograms(sb=sb,
                                                                                  sinkhorn_iteration=0,
                                                                                  device=device,
                                                                                  current_model=sb.training_model,
                                                                                  past_to_train_model=None)

    graph_metrics_and_paths_histograms(sb=sb,
                                       sinkhorn_iteration=0,
                                       device=device,
                                       current_model=sb.training_model,
                                       past_to_train_model=None,
                                       plot_path=None)