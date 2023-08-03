import os
import json
import torch
from graph_bridges.models.generative_models.sb import SB
from graph_bridges.models.backward_rates.backward_rate import GaussianTargetRateImageX0PredEMA
from graph_bridges.models.backward_rates.backward_rate_config import GaussianTargetRateImageX0PredEMAConfig


if __name__=="__main__":
    from graph_bridges.configs.graphs.config_sb import SBConfig
    from graph_bridges.data.graph_dataloaders_config import EgoConfig
    from graph_bridges.models.backward_rates.backward_rate_config import BackRateMLPConfig
    from graph_bridges.configs.graphs.config_sb import ParametrizedSamplerConfig

    config = SBConfig(experiment_indentifier="debug")
    config.data = EgoConfig(as_image=False, batch_size=32, full_adjacency=False)
    config.model = GaussianTargetRateImageX0PredEMAConfig(time_embed_dim=32, fix_logistic=False)
    config.sampler = ParametrizedSamplerConfig(num_steps=23)


    #read the model
    #config = get_config_from_file("graph", "lobster", "1687884918")
    device = torch.device(config.device)
    sb = SB(config,device)

    #test dataloaders
    databatch = next(sb.data_dataloader.train().__iter__())
    x_spins_data = databatch[0]
    number_of_paths = x_spins_data.shape[0]
    x_spins_noise = sb.target_dataloader.sample(number_of_paths,device)

    #test scheduler
    time_steps = sb.scheduler.set_timesteps(10,0.01,sinkhorn_iteration=1)
    print(time_steps.shape)

    # test model
    times = time_steps[6] * torch.ones(number_of_paths)
    generating_model : GaussianTargetRateImageX0PredEMA
    generating_model = sb.past_model
    forward_ = generating_model(x_spins_data.squeeze(),times)
    forward_stein = generating_model.stein_binary_forward(x_spins_data.squeeze(),times)


    # test losses
    estimator_ = sb.backward_ratio_stein_estimator.estimator(sb.training_model,
                                                             sb.past_model,
                                                             x_spins_data.squeeze(),
                                                             times)

    print(forward_.shape)
    print(forward_stein.shape)

    # test reference process
    x_spins_w_noise = sb.reference_process.spins_on_times(x_spins_data.squeeze(), times)


    # test pipeline
    print("From Dataloader image shape")
    x_end = sb.pipeline(None, 0, device, return_path=False)
    print(x_end.shape)

    print("From Dataloader full path in image shape with times")
    x_end, times = sb.pipeline(None, 0, device, return_path=True)
    print(x_end.shape)
    print(times.shape)

    print("From given start")
    x_end,times = sb.pipeline(None,0,device,x_spins_data,return_path=True)
    print(x_end.shape)
    print(times.shape)

    print("From given start in path shape")
    x_end,times = sb.pipeline(None,0,device,x_spins_data,return_path=True,return_path_shape=True)
    print(x_end.shape)
    print(times.shape)



