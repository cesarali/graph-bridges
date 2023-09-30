import torch
import unittest

from graph_bridges.models.generative_models.sb import SB
from graph_bridges.configs.config_sb import SBTrainerConfig
from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig
from graph_bridges.configs.config_sb import ParametrizedSamplerConfig, SteinSpinEstimatorConfig


from graph_bridges.models.trainers.sb_training import SBTrainer
from graph_bridges.models.schedulers.scheduling_sb import SBScheduler
from graph_bridges.data.dataloaders_utils import load_dataloader
from graph_bridges.configs.graphs.graph_config_sb import SBConfig

class TestSBScheduler(unittest.TestCase):

    sb_config:SBConfig
    sb:SB
    sb_trainer:SBTrainer

    def setUp(self) -> None:
        self.sb_config = SBConfig(delete=True,experiment_indentifier="unittest_sb_scheduler")
        self.sb_config.flip_estimator = SteinSpinEstimatorConfig(stein_sample_size=10)
        self.sb_config.sampler = ParametrizedSamplerConfig(num_steps=5)
        self.sb_config.trainer = SBTrainerConfig(learning_rate=1e-3,
                                                 num_epochs=6,
                                                 save_metric_epochs=2,
                                                 device="cuda:0",
                                                 metrics=["graphs_plots",
                                                        "histograms"])
        self.sb_config.initialize_new_experiment()

        self.device = torch.device("cpu")
        self.data_dataloader = load_dataloader(self.sb_config,type="data",device=self.device)
        self.target_dataloader = load_dataloader(self.sb_config,type="target",device=self.device)

        self.x_adj_target = next(self.target_dataloader.train().__iter__())[0].to(self.device)
        self.x_adj_data = next(self.data_dataloader.train().__iter__())[0].to(self.device)

        self.scheduler = SBScheduler(self.sb_config,device=self.device)

    def test_set_timesteps(self):
        timesteps0 = self.scheduler.set_timesteps(self.sb_config.sampler.num_steps,
                                                 self.sb_config.sampler.min_t,
                                                 sinkhorn_iteration=0)

        timesteps1 = self.scheduler.set_timesteps(self.sb_config.sampler.num_steps,
                                                 self.sb_config.sampler.min_t,
                                                 sinkhorn_iteration=1)

        timesteps2 = self.scheduler.set_timesteps(self.sb_config.sampler.num_steps,
                                                 self.sb_config.sampler.min_t,
                                                 sinkhorn_iteration=2)
        timesteps1_fliped = torch.flip(timesteps1,[0])

        self.assertTrue((timesteps0 == timesteps2).all())
        self.assertTrue((timesteps0 == timesteps1_fliped).all())

    def test_step(self):
        #h = self.select_time_difference(sinkhorn_iteration, timesteps, idx)
        #times = t * torch.ones(num_of_paths)
        #if sinkhorn_iteration != 0:
        #    logits = past_model.stein_binary_forward(initial_spins, times)
        #    rates_ = F.softplus(logits)
        #else:
        #    rates_ = self.reference_process.rates_states_and_times(initial_spins, times)
        return None

if __name__=="__main__":
    unittest.main()