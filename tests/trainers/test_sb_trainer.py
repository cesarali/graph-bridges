import torch
import unittest

from pprint import pprint


from graph_bridges.models.generative_models.sb import SB
from graph_bridges.configs.config_sb import SBTrainerConfig
from graph_bridges.data.graph_dataloaders_config import EgoConfig,CommunityConfig,CommunitySmallConfig
from graph_bridges.data.spin_glass_dataloaders_config import ParametrizedSpinGlassHamiltonianConfig
from graph_bridges.configs.graphs.graph_config_sb import SBConfig
from graph_bridges.configs.config_sb import ParametrizedSamplerConfig, SteinSpinEstimatorConfig

from graph_bridges.models.temporal_networks.mlp.temporal_mlp import TemporalMLPConfig,DeepTemporalMLPConfig
from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackwardRateTemporalHollowTransformerConfig
from graph_bridges.models.temporal_networks.transformers.temporal_hollow_transformers import TemporalHollowTransformerConfig
from graph_bridges.models.trainers.sb_training import SBTrainer
from graph_bridges.models.losses.loss_configs import RealFlipConfig
from graph_bridges.models.metrics.sb_metrics import  paths_marginal_histograms
from graph_bridges.models.reference_process.reference_process_config import GlauberDynamicsConfig
from graph_bridges.utils.plots.sb_plots import sinkhorn_plot

from graph_bridges.models.metrics.sb_paths_metrics import states_paths_histograms_plots

class TestSBTrainer(unittest.TestCase):

    sb:SB
    sb_config:SBConfig
    sb_trainer:SBTrainer

    def setUp(self) -> None:
        from graph_bridges.data.dataloaders_utils import check_sizes

        self.sb_config = SBConfig(delete=True,
                                  experiment_name="graph",
                                  experiment_type="sb",
                                  experiment_indentifier="unittest_sb_trainer")

        self.sb_config.temp_network = TemporalMLPConfig(time_embed_dim=10,hidden_dim=10)

        num_epochs = 2
        self.sb_config.flip_estimator = SteinSpinEstimatorConfig(stein_sample_size=10,
                                                                 stein_epsilon=0.2)

        self.sb_config.sampler = ParametrizedSamplerConfig(num_steps=5,
                                                           step_type="TauLeaping",
                                                           sample_from_reference_native=True)

        self.sb_config.trainer = SBTrainerConfig(learning_rate=0.001,
                                                 num_epochs=num_epochs,
                                                 save_metric_epochs=int(num_epochs),
                                                 save_model_epochs=int(num_epochs),
                                                 save_image_epochs=int(num_epochs),
                                                 clip_grad=False,
                                                 clip_max_norm=10.,
                                                 device="cuda:0",
                                                 metrics=["histograms"])

        self.sb_config.__post_init__()
        self.sb_trainer = SBTrainer(config=self.sb_config)

    def test_training(self):
        self.sb_trainer.train_schrodinger()

    @unittest.skip
    def test_loading(self):
        sb = SB()
        results,metrics,device = sb.load_from_results_folder(experiment_name="graph",
                                                             experiment_type="sb",
                                                             experiment_indentifier="community_small_to_bernoulli",
                                                             sinkhorn_iteration_to_load=0)
        current_model = sb.training_model
        past_to_train_model = None# sb.past_model
        #pprint(sb.config)
        sb.config.sampler.step_type = "TauLeaping"
        sinkhorn_iteration = 0

        all_histograms = paths_marginal_histograms(sb,
                                                   sinkhorn_iteration,
                                                   device,
                                                   current_model,
                                                   past_to_train_model,
                                                   exact_backward=True,
                                                   train=True)

        marginal_0, marginal_1, backward_histogram, forward_histogram, forward_time, state_legends = all_histograms

        sinkhorn_plot(sinkhorn_iteration,
                      marginal_0,
                      marginal_1,
                      backward_histogram=backward_histogram,
                      forward_histogram=forward_histogram,
                      time_=forward_time,
                      states_legends=state_legends)

    @unittest.skip
    def test_sinkhorn_initialization(self):
        current_model = self.sb_trainer.sb.training_model
        past_model = self.sb_trainer.sb.reference_process
        self.sb_trainer.initialize_sinkhorn(current_model,past_model,sinkhorn_iteration=0)

    def test_metrics_login(self):
        from graph_bridges.models.metrics.sb_metrics_utils import log_metrics
        results_dir = "C:/Users/cesar/Desktop/Projects/DiffusiveGenerativeModelling/Codes/sb_experiment"
        sb = SB()
        sinkhorn_iteration = 0
        results, metrics, device = sb.load_from_results_folder(experiment_dir=results_dir,
                                                               sinkhorn_iteration_to_load=sinkhorn_iteration)

        current_model = sb.training_model
        past_model = None
        log_metrics(sb,current_model,past_model,sinkhorn_iteration,"best",device,["mse_histograms"])

        print(metrics)

if __name__ == '__main__':
    unittest.main()
