import os
import torch
import numpy as np
import pandas as pd
from pprint import pprint
from dataclasses import asdict

if __name__=="__main__":
    from graph_bridges.data.dataloaders_utils import load_dataloader
    from graph_bridges.models.backward_rates.backward_rate_utils import load_backward_rates

    from graph_bridges.models.generative_models.ctdd import CTDD
    from graph_bridges.configs.graphs.graph_config_ctdd import CTDDConfig
    from graph_bridges.data.graph_dataloaders_config import EgoConfig, GraphSpinsDataLoaderConfig, TargetConfig
    from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig, GaussianTargetRateImageX0PredEMAConfig

    device = torch.device("cpu")

    config = CTDDConfig(experiment_indentifier="test_1")
    config.data = EgoConfig(as_image=False, batch_size=32, full_adjacency=False)
    config.model = GaussianTargetRateImageX0PredEMAConfig(time_embed_dim=32, fix_logistic=False)
    config.initialize_new_experiment()

    ctdd = CTDD()
    ctdd.create_new_from_config(config, device)

    minibatch = next(ctdd.data_dataloader.train().__iter__())
    x_adj = minibatch[0].to(device)
    x_features = minibatch[1].to(device)
    B = minibatch[0].shape[0]

    ts = torch.rand((B,), device=device) * (1.0 - config.loss.min_time) + config.loss.min_time
    x_t, x_tilde, qt0, rate = ctdd.scheduler.add_noise(x_adj,
                                                       ctdd.reference_process,
                                                       ts,
                                                       device,
                                                       return_dict=False)
    x_logits, p0t_reg, p0t_sig, reg_x = ctdd.model(minibatch, ts, x_tilde)
    loss_ = ctdd.loss(x_adj, x_tilde, qt0, rate, x_logits, reg_x, p0t_sig, p0t_reg, device)
    print(loss_)

