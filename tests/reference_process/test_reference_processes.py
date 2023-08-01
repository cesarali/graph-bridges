import torch

if __name__=="__main__":
    from graph_bridges.data.dataloaders_utils import load_dataloader
    from graph_bridges.configs.graphs.config_ctdd import CTDDConfig
    from graph_bridges.data.graph_dataloaders_config import EgoConfig, GraphSpinsDataLoaderConfig, TargetConfig
    from graph_bridges.models.backward_rates.backward_rate_config import BackRateMLPConfig, GaussianTargetRateImageX0PredEMAConfig
    from graph_bridges.models.reference_process.ctdd_reference import GaussianTargetRate

    device = torch.device("cpu")

    config = CTDDConfig(experiment_indentifier="test_1")
    config.data = EgoConfig(as_image=False, batch_size=5, full_adjacency=False)
    config.model = GaussianTargetRateImageX0PredEMAConfig(time_embed_dim=32, fix_logistic=False)
    config.initialize_new_experiment()

    data_dataloader = load_dataloader(config,type="data",device=device)
    target_dataloader = load_dataloader(config,type="target",device=device)

    x_adj_data = next(data_dataloader.train().__iter__())[0]
    x_adj_target = next(data_dataloader.train().__iter__())[0]
    batch_size = x_adj_data.shape[0]
    times = torch.rand(batch_size)

    reference_process = GaussianTargetRate(config,device)
    stein_binary_forward = reference_process.stein_binary_forward(states=x_adj_target,times=times)
    forward_ = reference_process.forward_rates(x_adj_data, times, device)
    rate_ = reference_process.rate(times)
    transition_ = reference_process.transition(times)
    forward_rates_and_probabilities_ = reference_process.forward_rates_and_probabilities(x_adj_data, times, device)
    forward_rates, qt0_denom, qt0_numer  = forward_rates_and_probabilities_
    forward_rates_and_probabilities_ = reference_process.forward_rates_and_probabilities(x_adj_target, times, device)
    forward_rates, qt0_denom, qt0_numer  = forward_rates_and_probabilities_
    forward_rates_ = reference_process.forward_rates(x_adj_data, times, device)
    forward_rates_ = reference_process.forward_rates(x_adj_target, times, device)

    print("rate_ {0}, transition_ {1}, forward_rates {2}, qt0_denom {3}, qt0_numer {4}, forward_rates_ {5}, ".format(rate_.shape,transition_.shape,forward_rates.shape,qt0_denom.shape,qt0_numer.shape,forward_rates_.shape))

