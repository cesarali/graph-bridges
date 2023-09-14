import torch
from graph_bridges.models.schedulers.scheduling_ctdd import CTDDScheduler

if __name__=="__main__":
    from graph_bridges.data.dataloaders_utils import load_dataloader
    from graph_bridges.models.backward_rates.backward_rate_utils import load_backward_rates

    from graph_bridges.models.generative_models.ctdd import CTDD
    from graph_bridges.configs.graphs.graph_config_ctdd import CTDDConfig
    from graph_bridges.data.graph_dataloaders_config import EgoConfig, GraphSpinsDataLoaderConfig, TargetConfig
    from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig, GaussianTargetRateImageX0PredEMAConfig
    from graph_bridges.models.reference_process.ctdd_reference import GaussianTargetRate

    device = torch.device("cuda:1")

    config = CTDDConfig(experiment_indentifier="test_1")
    config.data = EgoConfig(as_image=False, batch_size=5, full_adjacency=False)
    config.model = GaussianTargetRateImageX0PredEMAConfig(time_embed_dim=32, fix_logistic=False)
    config.initialize_new_experiment()

    data_dataloader = load_dataloader(config=config,type="data",device=device)
    target_dataloader = load_dataloader(config=config,type="target",device=device)
    model = load_backward_rates(config=config,device=device)
    reference_process = GaussianTargetRate(config,device)
    ctdd_scheduler = CTDDScheduler(config,device)

    minibatch = next(data_dataloader.train().__iter__())
    B = minibatch[0].shape[0]

    ts = torch.rand((B,), device=device) * (1.0 - config.loss.min_time) + config.loss.min_time
    x_t, x_tilde, qt0, rate = ctdd_scheduler.add_noise(minibatch[0],
                                                       reference_process,
                                                       ts,
                                                       device,
                                                       return_dict=False)
    x_logits, p0t_reg, p0t_sig, reg_x = model(minibatch, ts, x_tilde)

    print(x_t.shape)
