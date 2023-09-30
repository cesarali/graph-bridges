import os
import torch
import unittest
from graph_bridges.data.dataloaders_utils import load_dataloader
from graph_bridges.models.reference_process.reference_process_utils import load_reference

from graph_bridges.data.spin_glass_dataloaders_config import ParametrizedSpinGlassHamiltonianConfig
from graph_bridges.models.reference_process.reference_process_config import GlauberDynamicsConfig
from graph_bridges.configs.spin_glass.spin_glass_config_sb import SBConfig

from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig
from graph_bridges.models.temporal_networks.mlp.temporal_mlp import TemporalMLPConfig
from graph_bridges.models.reference_process.ctdd_reference import GaussianTargetRate
from graph_bridges.models.reference_process.glauber_reference import GlauberDynamics
from graph_bridges.models.schedulers.scheduling_sb import SBScheduler
from graph_bridges.models.generative_models.sb import SB



class TestGlauberDynamics(unittest.TestCase):

    def setUp(self) -> None:
        from graph_bridges.configs.spin_glass.spin_glass_config_sb import SBConfig

        self.sb_config = SBConfig(experiment_indentifier="test_glauber")

        self.sb_config.data = ParametrizedSpinGlassHamiltonianConfig(data="bernoulli_probability_0.2")
        self.sb_config.target = ParametrizedSpinGlassHamiltonianConfig(data="bernoulli_probability_0.9")

        self.sb_config.temp_network = TemporalMLPConfig()
        self.sb_config.reference = GlauberDynamicsConfig()
        self.sb_config.initialize_new_experiment()

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.sb = SB()
        self.sb.create_new_from_config(self.sb_config,self.device)

        # obtain dataloaders
        self.sb.data_dataloader = load_dataloader(self.sb_config, type="data", device=self.device)
        self.sb.target_dataloader = load_dataloader(self.sb_config, type="target", device=self.device)

        # obtain data
        self.x_adj_data = next(self.sb.data_dataloader.train().__iter__())[0].to(self.device)
        self.x_adj_target = next(self.sb.target_dataloader.train().__iter__())[0].to(self.device)
        self.batch_size = self.x_adj_data.shape[0]
        self.times = torch.rand(self.batch_size,device=self.device)

    @unittest.skip
    def test_sampling(self):
        self.sb.scheduler.set_timesteps(self.sb_config.sampler.num_steps,
                                        self.sb_config.sampler.min_t,
                                        sinkhorn_iteration=0)
        timesteps_ = self.sb.scheduler.timesteps
        paths, time_steps = self.sb.reference_process.sample_path(self.x_adj_data,timesteps_)
        print(f"Paths Shape {paths.shape}")
        print(f"times_steps {time_steps.shape}")

    @unittest.skip
    def test_rates_and_probabilities(self):
        i_range = torch.full((self.sb_config.data.batch_size,), 2).to(self.device)
        rates_ = self.sb.reference_process.selected_flip_rates(self.x_adj_data, i_range)
        print(f"Selected Rates shape {rates_.shape}")
        all_flip_rates = self.sb.reference_process.all_flip_rates(self.x_adj_data)
        print(f"All Flip Rates {all_flip_rates.shape}")
        transition_rates = self.sb.reference_process.transition_rates_states(self.x_adj_data)
        print(f"All Flip Rates {transition_rates.shape}")

    @unittest.skip
    def test_pipelines(self):
        from graph_bridges.models.spin_glass.spin_states_statistics import spin_states_stats

        """
        paths, times_ = self.sb.pipeline(generation_model=None,
                                         sinkhorn_iteration=0,
                                         device=self.device,
                                         initial_spins=self.x_adj_data,
                                         sample_from_reference_native=True,
                                         return_path=True,
                                         return_path_shape=False)
        print(f"Paths Shape {paths.shape}")
        print(f"Times Shape {times_.shape}")
        
        """

        """
        paths, times_ = self.sb.pipeline(generation_model=None,
                                         sinkhorn_iteration=0,
                                         device=self.device,
                                         initial_spins=self.x_adj_data,
                                         sample_from_reference_native=False,
                                         return_path=True,
                                         return_path_shape=True)
        print(f"Paths Shape {paths.shape}")
        print(f"Times Shape {times_.shape}")
        """

        for databatch in self.sb.pipeline.paths_iterator(generation_model=None,
                                                         sinkhorn_iteration=0,
                                                         device=self.device,
                                                         train=True,
                                                         return_path=True,
                                                         return_path_shape=True):
            print(f"paths {databatch[0].shape}")
            print(f"time {databatch[1].shape}")

        #stats_ = spin_states_stats(self.sb_config.data.number_of_spins)
        #time_series_of_paths = stats_.counts_states_in_paths(paths.cpu())
        #print(time_series_of_paths.shape)

    def test_losses(self):
        current_model = self.sb.training_model
        past_model = self.sb.reference_process

        paths, times_ = self.sb.pipeline(generation_model=None,
                                         sinkhorn_iteration=0,
                                         device=self.device,
                                         initial_spins=self.x_adj_data,
                                         sample_from_reference_native=True,
                                         return_path=True,
                                         return_path_shape=False)
        print(f"Paths Shape {paths.shape}")
        print(f"Times Shape {times_.shape}")
        loss_ = self.sb.backward_ratio_estimator.__call__(current_model, past_model, paths, times_)
        print(loss_.shape)

    def test_metrics(self):
        from graph_bridges.models.metrics.sb_paths_metrics import states_paths_histograms_plots

        current_model = self.sb.training_model
        past_model = 0


        states_paths_histograms_plots(self.sb,
                                      sinkhorn_iteration=0,
                                      device=self.device,
                                      current_model=current_model,
                                      past_to_train_model=past_model,
                                      plot_path=None)




if __name__=="__main__":
    unittest.main()
