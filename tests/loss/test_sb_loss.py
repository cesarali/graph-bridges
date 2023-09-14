import os
import sys
import torch
from pathlib import Path
from graph_bridges import results_path
from graph_bridges.models.backward_rates.ctdd_backward_rate import BackRateConstant
from graph_bridges.models.backward_rates.ctdd_backward_rate import GaussianTargetRateImageX0PredEMA
from graph_bridges.models.backward_rates.ctdd_backward_rate_config import GaussianTargetRateImageX0PredEMAConfig
from graph_bridges.models.generative_models.sb import SB

results_path = Path(results_path)
loss_study_path = results_path / "graph" / "lobster" / "contant_past_model_loss.json"

if __name__=="__main__":
    from graph_bridges.configs.graphs.graph_config_sb import SBConfig
    from graph_bridges.data.graph_dataloaders_config import EgoConfig
    from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig

    config = SBConfig(experiment_indentifier="debug")
    config.data = EgoConfig(as_image=False, batch_size=32, full_adjacency=False)
    config.model = GaussianTargetRateImageX0PredEMAConfig(time_embed_dim=32, fix_logistic=False)

    #read the model
    device = torch.device(config.device)
    sb = SB(config,device)

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
    estimator_ = sb.backward_ratio_stein_estimator.estimator(sb.training_model,
                                                             sb.past_model,
                                                             x_spins_data.squeeze(),
                                                             times)


"""
from graph_bridges.models.backward_rates.backward_rate import BackRateConstant
from pathlib import Path
from graph_bridges import results_path
results_path = Path(results_path)
loss_study_path = results_path / "graph" / "lobster" / "contant_past_model_loss.json"


# FULL AVERAGE
for spins_path, times in sb.pipeline.paths_iterator(None, sinkhorn_iteration=0):
    loss = sb.backward_ration_stein_estimator.estimator(sb.training_model,
                                                        past_constant,
                                                        spins_path,
                                                        times)
    print(loss)
    break

contant_error = {}
for constant_ in [0.1,1.,10.,100.]:
    past_constant = BackRateConstant(config,device,None,constant_)
    # PER TIME
    error_per_timestep = {}
    for spins_path, times in sb.pipeline.paths_iterator(None, sinkhorn_iteration=0,return_path=True,return_path_shape=True):
        total_times_steps = times.shape[-1]
        for t in range(total_times_steps):
            spins_ = spins_path[:,t,:]
            times_ = times[:,t]
            loss = sb.backward_ration_stein_estimator.estimator(sb.training_model,
                                                                past_constant,
                                                                spins_,
                                                                times_)
            try:
                error_per_timestep[t].append(loss.item())
            except:
                error_per_timestep[t] = [loss.item()]
    contant_error[constant_] = error_per_timestep


json.dump(contant_error,open(loss_study_path,"w"))
print(contant_error)
"""
"""
times_batch_1 = []
paths_batch_1 = []
for spins_path, times in sb.pipeline.paths_iterator(training_model, sinkhorn_iteration=sinkhorn_iteration + 1):
    paths_batch_1.append(spins_path)
    times_batch_1.append(times)
"""
# test plots

"""
sinkhorn_plot(sinkhorn_iteration=0,
              states_histogram_at_0=0,
              states_histogram_at_1=0,
              backward_histogram=0,
              forward_histogram=0,
              time_=None,
              states_legends=0)
--"""
