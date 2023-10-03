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

from graph_bridges.models.metrics.sb_metrics import marginal_paths_histograms_plots, paths_marginal_histograms
from graph_bridges.models.reference_process.reference_process_config import GlauberDynamicsConfig
from graph_bridges.models.trainers.sb_training import SBTrainer
from graph_bridges.models.losses.loss_configs import RealFlipConfig

from graph_bridges.models.metrics.sb_paths_metrics import states_paths_histograms_plots
from graph_bridges.models.metrics.sb_metrics import marginal_paths_histograms_plots
from graph_bridges.data.dataloaders_utils import check_sizes


def test_loading(sb):
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
    pprint(sb_config.data.__dict__)
    print("Loaded")
    pprint(sb.config.data.__dict__)
    """


if __name__ == '__main__':
    sb_config = SBConfig(delete=True,
                              experiment_name="graph",
                              experiment_type="sb",
                              experiment_indentifier="community_small_equal_time_and_data")
    # experiment_indentifier="bernoulli_to_bernoulli_0_stein_200_mlp_lr01_steps_50_clip_False_gradient_exact")

    #=====================================
    # DATA
    #=====================================

    sb_config.data = CommunitySmallConfig(as_image=False,
                                          batch_size=25,
                                               full_adjacency=False)

    # sb_config.data = ParametrizedSpinGlassHamiltonianConfig(data="bernoulli_probability_test",
    #                                                             bernoulli_spins= True,
    #                                                            bernoulli_probability=0.25,
    #                                                             delete_data=True,
    #                                                             number_of_paths=250,
    #                                                             number_of_spins=25,
    #                                                             batch_size=20)

    # sb_config.data = CommunityConfig(as_image=False, batch_size=batch_size, full_adjacency=True)

    data_sizes = check_sizes(sb_config)
    print(data_sizes)

    # sb_config.data = ParametrizedSpinGlassHamiltonianConfig(data="bernoulli_probability_0.2",
    #                                                             batch_size=10)

    #=====================================
    # TARGET
    #=====================================

    # sb_config.target = ParametrizedSpinGlassHamiltonianConfig(data="bernoulli_probability_0.9",
    #                                                               batch_size=10)

    # sb_config.target = ParametrizedSpinGlassHamiltonianConfig(number_of_paths=data_sizes.total_data_size,
    #                                                               number_of_spins=data_sizes.D,
    #                                                               data="bernoulli_probability_test_target",
    #                                                               delete_data=True,
    #                                                               bernoulli_spins=True,
    #                                                               bernoulli_probability=0.75)


    #=====================================
    # ARCHITECTURES
    #=====================================

    sb_config.temp_network = TemporalMLPConfig(time_embed_dim=250,hidden_dim=250)

    # sb_config.temp_network = DeepTemporalMLPConfig(layers_dim=[100,300],
    #                                                    time_embed_dim=200,
    #                                                    time_embed_hidden=100,
    #                                                    data_hidden=50,
    #                                                    dropout=.25)#

    # sb_config.reference = GlauberDynamicsConfig(fom_data_hamiltonian=False,
    #                                                 gamma=10.)

    #==================================
    # LOSS
    #==================================

    num_epochs = 300
    sb_config.flip_estimator = SteinSpinEstimatorConfig(stein_sample_size=200,
                                                             stein_epsilon=0.2)
    # sb_config.flip_estimator = RealFlipConfig()

    sb_config.sampler = ParametrizedSamplerConfig(num_steps=40,
                                                       step_type="TauLeaping",
                                                       sample_from_reference_native=True)

    sb_config.trainer = SBTrainerConfig(learning_rate=0.01,
                                             num_epochs=num_epochs,
                                             save_metric_epochs=int(num_epochs * .25),
                                             save_model_epochs=int(num_epochs * .5),
                                             save_image_epochs=int(num_epochs * .5),
                                             clip_grad=False,
                                             clip_max_norm=10.,
                                             device="cuda:0",
                                             metrics=["histograms"])

    # ["graphs_plots", "histograms"]
    sb_config.__post_init__()
    sb_trainer = SBTrainer(config=sb_config)
    sb_trainer.train_schrodinger()
