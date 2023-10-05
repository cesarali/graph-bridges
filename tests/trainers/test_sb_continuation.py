import torch
import unittest

from pprint import pprint


from graph_bridges.models.generative_models.sb import SB
from graph_bridges.configs.config_sb import SBTrainerConfig
from graph_bridges.data.graph_dataloaders_config import EgoConfig,CommunityConfig
from graph_bridges.data.spin_glass_dataloaders_config import ParametrizedSpinGlassHamiltonianConfig
from graph_bridges.configs.graphs.graph_config_sb import SBConfig
from graph_bridges.configs.config_sb import ParametrizedSamplerConfig, SteinSpinEstimatorConfig
from graph_bridges.models.metrics.sb_metrics import marginal_paths_histograms_plots

from graph_bridges.models.temporal_networks.mlp.temporal_mlp import TemporalMLPConfig,DeepTemporalMLPConfig
from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackwardRateTemporalHollowTransformerConfig
from graph_bridges.models.temporal_networks.transformers.temporal_hollow_transformers import TemporalHollowTransformerConfig
from graph_bridges.models.trainers.sb_training import SBTrainer
from graph_bridges.models.losses.loss_configs import RealFlipConfig
from graph_bridges.models.metrics.sb_metrics import  paths_marginal_histograms
from graph_bridges.models.reference_process.reference_process_config import GlauberDynamicsConfig


from graph_bridges.models.metrics.sb_paths_metrics import states_paths_histograms_plots
from graph_bridges.utils.test_utils import check_model_devices

class TestSBTrainer(unittest.TestCase):

    sb:SB
    sb_config:SBConfig
    sb_trainer:SBTrainer

    @unittest.skip
    def test_loading(self):
        sb = SB()
        sb.load_from_results_folder(experiment_name="graph",
                                    experiment_type="sb",
                                    experiment_indentifier="1696030913",
                                    new_experiment=False,
                                    new_experiment_indentifier="Harz2",
                                    sinkhorn_iteration_to_load=0,
                                    checkpoint=1500,
                                    device=torch.device("cpu"))

        current_model = sb.training_model
        past_model = sb.past_model
        sb.config.sampler.step_type = "poisson"

    @unittest.skip
    def test_restart_training(self):
        sb_trainer = SBTrainer(config=None,
                               experiment_name="graph",
                               experiment_type="sb",
                               experiment_indentifier="community_small_to_bernoulli",
                               new_experiment_indentifier="community_small_to_bernoulli_reverse_pipeline_5",
                               sinkhorn_iteration_to_load=0,
                               next_sinkhorn=True)
        sb_trainer.sb_config.trainer.save_metric_epochs = 100
        sb_trainer.number_of_epochs = 1500
        sb_trainer.train_schrodinger()

    def test_load_and_plot(self):
        device = torch.device("cpu")

        sb = SB()
        sb.load_from_results_folder(experiment_name="graph",
                                    experiment_type="sb",
                                    experiment_indentifier="community_small_to_bernoulli",
                                    new_experiment=False,
                                    new_experiment_indentifier=None,
                                    sinkhorn_iteration_to_load=0,
                                    device=device)

        current_model = sb.training_model
        past_model = sb.past_model
        #past_model.load_state_dict(current_model.state_dict())

        marginal_paths_histograms_plots(sb,
                                        sinkhorn_iteration=0,
                                        device=device,
                                        current_model=current_model,
                                        past_to_train_model=None,
                                        save_path=None,
                                        exact_backward=False,
                                        train=True)

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
