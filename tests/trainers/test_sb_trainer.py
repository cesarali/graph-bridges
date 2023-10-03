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
from graph_bridges.models.metrics.sb_metrics import marginal_paths_histograms_plots, paths_marginal_histograms
from graph_bridges.models.reference_process.reference_process_config import GlauberDynamicsConfig


from graph_bridges.models.metrics.sb_paths_metrics import states_paths_histograms_plots
from graph_bridges.models.metrics.sb_metrics import marginal_paths_histograms_plots

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
        sb.load_from_results_folder(experiment_name="graph",
                                    experiment_type="sb",
                                    experiment_indentifier="1695991977",
                                    sinkhorn_iteration_to_load=0)
        current_model = sb.training_model
        past_model = sb.past_model
        pprint(sb.config)
        sb.config.sampler.step_type = "TauLeaping"
        marginal_paths_histograms_plots(sb,
                                        sinkhorn_iteration=0,
                                        device=sb.training_model.parameters().__next__().device,
                                        current_model=current_model,
                                        past_to_train_model=None,
                                        plot_path=None)
        """
        if hasattr(sb.data_dataloader,"transform_to_graph"):
            x_adj = sb.data_dataloader,"transform_to_graph"(x_end)
        print("Original")
        pprint(self.sb_config.data.__dict__)
        print("Loaded")
        pprint(sb.config.data.__dict__)
        """

    @unittest.skip
    def test_sinkhorn_initialization(self):
        current_model = self.sb_trainer.sb.training_model
        past_model = self.sb_trainer.sb.reference_process
        self.sb_trainer.initialize_sinkhorn(current_model,past_model,sinkhorn_iteration=0)

    @unittest.skip
    def test_metrics_login(self):
        sinkhorn_iteration = 0
        current_model = self.sb_trainer.sb.training_model
        past_model = self.sb_trainer.sb.reference_process

        self.sb_trainer.log_metrics(current_model=current_model,
                                    past_to_train_model=past_model,
                                    sinkhorn_iteration=sinkhorn_iteration,
                                    epoch=10,
                                    device=self.sb_trainer.device)

if __name__ == '__main__':
    unittest.main()
