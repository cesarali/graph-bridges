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
    x_spins_data = databatch[0]
    number_of_paths = x_spins_data.shape[0]
    x_spins_noise = sb.target_dataloader.sample(number_of_paths,device)

    #test scheduler
    time_steps = sb.scheduler.set_timesteps(10,0.01,sinkhorn_iteration=1)
    print(time_steps)

    # test model
    times = time_steps[6] * torch.ones(number_of_paths)
    generating_model : GaussianTargetRateImageX0PredEMA
    generating_model = sb.past_model
    forward_ = generating_model(x_spins_data.squeeze(),times)
    forward_stein = generating_model.stein_binary_forward(x_spins_data.squeeze(),times)

    # test losses
    from graph_bridges.models.losses.estimators import BackwardRatioSteinEstimator
    backward_ration_stein_estimator = BackwardRatioSteinEstimator(config,device)
    estimator_ = backward_ration_stein_estimator.estimator(sb.training_model,
                                                           sb.past_model,
                                                           x_spins_data.squeeze(),
                                                           times)

    #print(forward_.shape)
    #print(forward_stein.shape)

    # stuff
    sb.pipeline.bridge_config.sampler.num_steps = 20
    # test reference process
    x_spins_w_noise = sb.reference_process.spins_on_times(x_spins_data.squeeze(), times)

    # test pipeline
    print("Hey!")
    x_end,times = sb.pipeline(None,0,device,x_spins_data,return_path=True)
    print(x_end.shape)
    print(times.shape)

    print("Hey 2!")
    x_end,times = sb.pipeline(None,0,device,x_spins_data,return_path=True,return_path_shape=True)
    print(x_end.shape)
    print(times.shape)

    print("Go!")
    # test trainer
    check_training_path = True
    sinkhorn_iteration = 0
    if sinkhorn_iteration == 0:
        past_model = None
    else:
        past_model = sb.past_model
    training_model = sb.training_model

    for spins_path, times in sb.pipeline.paths_iterator(past_model, sinkhorn_iteration=sinkhorn_iteration,return_path_shape=True):
        if check_training_path:
            end_of_path = spins_path[:, -1, :].unsqueeze(1).unsqueeze(1)
            x_end, times = sb.pipeline(training_model,
                                       sinkhorn_iteration+1,
                                       device,
                                       end_of_path,
                                       return_path=True,
                                       return_path_shape=True)

        print(end_of_path.shape)
        print(x_spins_data.shape)
        break

    for spins_path, times in sb.pipeline.paths_iterator(past_model, sinkhorn_iteration=sinkhorn_iteration):
        loss = backward_ration_stein_estimator.estimator(sb.training_model,
                                                         sb.past_model,
                                                         spins_path,
                                                         times)
        print(loss)
        break

    """
    times_batch_1 = []
    paths_batch_1 = []
    for spins_path, times in sb.pipeline.paths_iterator(training_model, sinkhorn_iteration=sinkhorn_iteration + 1):
        paths_batch_1.append(spins_path)
        times_batch_1.append(times)
    """
    # test plots
    from graph_bridges.utils.plots.sb_plots import sinkhorn_plot

    """
    sinkhorn_plot(sinkhorn_iteration=0,
                  states_histogram_at_0=0,
                  states_histogram_at_1=0,
                  backward_histogram=0,
                  forward_histogram=0,
                  time_=None,
                  states_legends=0)
    --"""

    #test trainer



