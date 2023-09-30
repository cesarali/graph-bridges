import torch
import unittest

from pprint import pprint


from graph_bridges.models.generative_models.sb import SB
from graph_bridges.configs.config_sb import SBTrainerConfig
from graph_bridges.data.graph_dataloaders_config import EgoConfig
from graph_bridges.data.spin_glass_dataloaders_config import ParametrizedSpinGlassHamiltonianConfig
from graph_bridges.configs.graphs.graph_config_sb import SBConfig
from graph_bridges.configs.config_sb import ParametrizedSamplerConfig, SteinSpinEstimatorConfig

from graph_bridges.models.temporal_networks.mlp.temporal_mlp import TemporalMLPConfig
from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackwardRateTemporalHollowTransformerConfig
from graph_bridges.models.temporal_networks.transformers.temporal_hollow_transformers import TemporalHollowTransformerConfig
from graph_bridges.models.trainers.sb_training import SBTrainer
from graph_bridges.models.spin_glass.spin_utils import copy_and_flip_spins
from graph_bridges.models.losses.loss_configs import RealFlipConfig
from graph_bridges.models.losses.estimators import (
    RealFlip,
    GradientFlipEstimator
)

class TestSBLoss(unittest.TestCase):

    def test_stein_flip(self):
        from graph_bridges.data.transforms import SpinsToBinaryTensor
        from graph_bridges.models.losses.loss_configs import GradientEstimatorConfig

        spins_to_binary = SpinsToBinaryTensor()
        self.sb_config = SBConfig(delete=True,
                                  experiment_name="spin_glass",
                                  experiment_type="sb",
                                  experiment_indentifier=None)

        batch_size = 2
        self.sb_config.data = ParametrizedSpinGlassHamiltonianConfig(data="bernoulli_small",
                                                                     batch_size=batch_size,
                                                                     number_of_spins=3)
        self.sb_config.target = ParametrizedSpinGlassHamiltonianConfig(data="bernoulli_small",
                                                                       batch_size=batch_size,
                                                                       number_of_spins=3)
        self.sb_config.temp_network = TemporalMLPConfig(time_embed_dim=12,hidden_dim=250)

        #self.sb_config.flip_estimator = SteinSpinEstimatorConfig(stein_sample_size=200, stein_epsilon=0.2)
        self.sb_config.flip_estimator = GradientEstimatorConfig()
        self.sb_config.flip_estimator = RealFlipConfig()

        self.sb_config.sampler = ParametrizedSamplerConfig(num_steps=20)
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        sb = SB()
        sb.create_new_from_config(self.sb_config,self.device)
        real_flip_estimator = RealFlip()

        batchdata = next(sb.data_dataloader.train().__iter__())
        X_spins =  batchdata[0].to(self.device)

        current_time = torch.rand((batch_size)).to(self.device)

        past_to_train = sb.reference_process
        current_model = sb.training_model

        flip_estimate_ = sb.backward_ratio_estimator.flip_estimator(current_model, X_spins, current_time)
        real_flip = real_flip_estimator(current_model, X_spins, current_time)
        loss = sb.backward_ratio_estimator(current_model,past_to_train,X_spins,current_time)

        print("flip_estimate")
        print(flip_estimate_)
        print("real flip")
        print(real_flip)
        print("loss")
        print(loss)


if __name__=="__main__":
    unittest.main()

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
