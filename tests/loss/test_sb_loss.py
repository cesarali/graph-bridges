import os
import sys
from pathlib import Path
from graph_bridges import results_path
from graph_bridges.models.backward_rates.backward_rate import BackRateConstant

results_path = Path(results_path)
loss_study_path = results_path / "graph" / "lobster" / "contant_past_model_loss.json"

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
