import torch
import unittest
from graph_bridges.data.dataloaders_utils import load_dataloader
from graph_bridges.models.reference_process.reference_process_utils import load_reference

from graph_bridges.configs.spin_glass.spin_glass_config_sb import SBConfig
from graph_bridges.data.spin_glass_dataloaders_config import ParametrizedSpinGlassHamiltonianConfig
from graph_bridges.models.reference_process.reference_process_config import GlauberDynamicsConfig

from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig
from graph_bridges.models.reference_process.ctdd_reference import GaussianTargetRate
from graph_bridges.models.reference_process.glauber_reference import GlauberDynamics
from graph_bridges.models.schedulers.scheduling_sb import SBScheduler

class TestGlauberDynamics(unittest.TestCase):

    def setUp(self) -> None:
        from graph_bridges.configs.spin_glass.spin_glass_config_sb import SBConfig

        self.sb_config = SBConfig(experiment_indentifier="test_glauber")
        self.sb_config.data = ParametrizedSpinGlassHamiltonianConfig(batch_size=23)
        self.sb_config.reference = GlauberDynamicsConfig()
        self.sb_config.initialize_new_experiment()

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # obtain dataloaders
        self.data_dataloader = load_dataloader(self.sb_config, type="data", device=self.device)
        self.target_dataloader = load_dataloader(self.sb_config, type="target", device=self.device)

        # obtain data
        self.x_adj_data = next(self.data_dataloader.train().__iter__())[0].to(self.device)
        self.x_adj_target = next(self.target_dataloader.train().__iter__())[0].to(self.device)
        self.batch_size = self.x_adj_data.shape[0]
        self.times = torch.rand(self.batch_size,device=self.device)

        self.reference_process = GlauberDynamics(self.sb_config, self.device)
        self.scheduler = SBScheduler(self.sb_config, self.device)

    def test_sampling(self):
        self.scheduler.set_timesteps(self.sb_config.sampler.num_steps,
                                     self.sb_config.sampler.min_t,
                                     sinkhorn_iteration=0)
        timesteps_ = self.scheduler.timesteps
        paths, time_steps = self.reference_process.sample_path(self.x_adj_data,timesteps_)
        print(f"Paths Shape {paths.shape}")
        print(f"times_steps {time_steps.shape}")

    def test_rates_and_probabilities(self):
        i_range = torch.full((self.sb_config.data.batch_size,), 2).to(self.device)
        rates_ = self.reference_process.selected_flip_rates(self.x_adj_data, i_range)
        print(f"Selected Rates shape {rates_.shape}")
        all_flip_rates = self.reference_process.all_flip_rates(self.x_adj_data)
        print(f"All Flip Rates {all_flip_rates.shape}")
        transition_rates = self.reference_process.transition_rates_states(self.x_adj_data)
        print(f"All Flip Rates {transition_rates.shape}")






if __name__=="__main__":
    unittest.main()
