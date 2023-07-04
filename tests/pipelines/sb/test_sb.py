import os
import torch
import unittest
from graph_bridges.models.generative_models.sb import SB
from graph_bridges.configs.graphs.lobster.config_base import get_config_from_file
from graph_bridges.models.backward_rates.backward_rate import GaussianTargetRateImageX0PredEMA


if __name__=="__main__":

    #read the model
    config = get_config_from_file("graph", "lobster", "1687884918")
    device = torch.device(config.device)
    sb = SB()
    sb.create_from_config(config, device)

    #test dataloaders
    databatch = next(sb.data_dataloader.train().__iter__())
    x_spins = databatch[0]

    #test scheduler
    time_steps = sb.scheduler.set_timesteps(10,0.01,sinkhorn_iteration=1)
    print(time_steps)

    # test model
    times = time_steps[6] * torch.ones(x_spins.shape[0])
    generating_model : GaussianTargetRateImageX0PredEMA
    generating_model = sb.past_model
    forward_ = generating_model(x_spins.squeeze(),times)
    forward_stein = generating_model.stein_binary_forward(x_spins.squeeze(),times)

    # test losses
    from graph_bridges.models.losses.estimators import BackwardRatioSteinEstimator

    backward_ration_stein_estimator = BackwardRatioSteinEstimator(config,device)

    #print(forward_.shape)
    #print(forward_stein.shape)
    # stuff

    #test pipeline
    x = sb.pipeline(sb.past_model,sinkhorn_iteration=1)
    #print(x.shape)
