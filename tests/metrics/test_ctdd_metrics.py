import torch
from graph_bridges.models.metrics.ctdd_metrics import graph_metrics_for_ctdd, marginal_histograms_for_ctdd

if __name__=="__main__":


    from graph_bridges.models.generative_models.ctdd import CTDD
    from graph_bridges.configs.graphs.config_ctdd import CTDDConfig
    from graph_bridges.data.graph_dataloaders_config import EgoConfig, CommunityConfig, GraphSpinsDataLoaderConfig, TargetConfig
    from graph_bridges.models.backward_rates.backward_rate_config import BackRateMLPConfig, GaussianTargetRateImageX0PredEMAConfig

    device = torch.device("cpu")

    config = CTDDConfig(experiment_indentifier="test_1")
    config.data = EgoConfig(as_image=False, batch_size=32, full_adjacency=False)
    config.model = GaussianTargetRateImageX0PredEMAConfig(time_embed_dim=32, fix_logistic=False)
    config.initialize_new_experiment()

    ctdd = CTDD()
    ctdd.create_new_from_config(config, device)

    graph_metrics = graph_metrics_for_ctdd(ctdd,config)
    #marginal_histograms = marginal_histograms_for_ctdd(ctdd,config,device)

