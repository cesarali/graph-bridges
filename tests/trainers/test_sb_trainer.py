import torch
import unittest

from pprint import pprint


from graph_bridges.models.generative_models.sb import SB
from graph_bridges.configs.config_sb import SBTrainerConfig
from graph_bridges.data.graph_dataloaders_config import EgoConfig,CommunityConfig
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
                                  experiment_indentifier=None)

        self.sb_config.data = EgoConfig(as_image=False,
                                        batch_size=25,
                                        full_adjacency=False)

        #self.sb_config.data = ParametrizedSpinGlassHamiltonianConfig(data="bernoulli_probability_test",
        #                                                             bernoulli_spins= True,
        #                                                             bernoulli_probability=0.85,
        #                                                             delete_data=True,
        #                                                             number_of_paths = 300,
        #                                                             number_of_spins=4,
        #                                                             batch_size=32)

        #self.sb_config.data = CommunityConfig(as_image=False, batch_size=self.batch_size, full_adjacency=True)

        data_sizes = check_sizes(self.sb_config)
        self.sb_config.target = ParametrizedSpinGlassHamiltonianConfig(number_of_paths=data_sizes.total_data_size,
                                                                       number_of_spins=data_sizes.D,
                                                                       data="graph_bernoulli",
                                                                       delete_data=True,
                                                                       bernoulli_spins=True,
                                                                       bernoulli_probability=0.84)

        #self.sb_config.data = ParametrizedSpinGlassHamiltonianConfig(data="bernoulli_probability_0.2",
        #                                                             batch_size=10)
        #self.sb_config.target = ParametrizedSpinGlassHamiltonianConfig(data="bernoulli_probability_0.9",
        #                                                               batch_size=10)

        self.sb_config.temp_network = TemporalMLPConfig(time_embed_dim=159,hidden_dim=650)

        #self.sb_config.temp_network = DeepTemporalMLPConfig(layers_dim=[500,200],
        #                                                    time_embed_dim=200,
        #                                                    time_embed_hidden=300,
        #                                                    dropout=.5)

        #self.sb_config.reference = GlauberDynamicsConfig(fom_data_hamiltonian=False,
        #                                                 gamma=10.)

        num_epochs = 3000
        self.sb_config.flip_estimator = SteinSpinEstimatorConfig(stein_sample_size=200,
                                                                 stein_epsilon=0.2)
        #self.sb_config.flip_estimator = RealFlipConfig()

        self.sb_config.sampler = ParametrizedSamplerConfig(num_steps=40,
                                                           step_type="TauLeaping",
                                                           sample_from_reference_native=True)

        self.sb_config.trainer = SBTrainerConfig(learning_rate=0.007,
                                                 num_epochs=num_epochs,
                                                 save_metric_epochs=10,
                                                 save_model_epochs=int(num_epochs*.5),
                                                 save_image_epochs=int(num_epochs*.5),
                                                 clip_grad=True,
                                                 clip_max_norm=1.,
                                                 device="cuda:0",
                                                 metrics=["histograms","paths_histograms"])
        #["graphs_plots", "histograms"]
        self.sb_config.__post_init__()
        self.sb_trainer = SBTrainer(self.sb_config)

    @unittest.skip
    def test_training(self):
        self.sb_trainer.train_schrodinger()

        self.sb = self.sb_trainer.sb
        current_model = self.sb_trainer.sb.training_model

       #past_model = None

        paths, times_ = self.sb.pipeline(generation_model=None,
                                         sinkhorn_iteration=0,
                                         device=self.sb_trainer.device,
                                         initial_spins=None,
                                         sample_from_reference_native=True,
                                         return_path=True,
                                         return_path_shape=False)

        #marginal_paths_histograms_plots(self.sb,
        #                                sinkhorn_iteration=0,
        #                                device=self.sb.training_model.parameters().__next__().device,
        #                                current_model=current_model,
        #                                past_to_train_model=None,
        #                                plot_path=None)




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
