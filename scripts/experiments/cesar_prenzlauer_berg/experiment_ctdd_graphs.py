from graph_bridges.models.trainers.ctdd_training import CTDDTrainer

if __name__=="__main__":
    from graph_bridges.configs.graphs.graph_config_ctdd import CTDDConfig
    from graph_bridges.configs.config_ctdd import CTDDTrainerConfig,ParametrizedSamplerConfig

    from graph_bridges.data.graph_dataloaders_config import EgoConfig,CommunityConfig,GridConfig,CommunitySmallConfig
    from graph_bridges.data.graph_dataloaders_config import CommunitySmallConfig

    from graph_bridges.models.temporal_networks.mlp.temporal_mlp import TemporalMLPConfig
    from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig
    from graph_bridges.models.temporal_networks.convnets.autoencoder import ConvNetAutoencoderConfig
    from graph_bridges.models.backward_rates.ctdd_backward_rate_config import GaussianTargetRateImageX0PredEMAConfig
    from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackwardRateTemporalHollowTransformerConfig
    from graph_bridges.models.temporal_networks.transformers.temporal_hollow_transformers import TemporalHollowTransformerConfig

    ctdd_config = CTDDConfig(experiment_indentifier="mlp_test_5_community",
                             experiment_name="graph",
                             experiment_type="ctdd")

    #ctdd_config.data = EgoConfig(batch_size=24,
    #                             full_adjacency=True)

    ctdd_config.data = CommunitySmallConfig(batch_size=24,
                                       full_adjacency=True)

    # MLP
    ctdd_config.model = BackRateMLPConfig()
    #ctdd_config.temp_network = TemporalMLPConfig()

    # CONVNET
    #ctdd_config.model = GaussianTargetRateImageX0PredEMAConfig()
    #ctdd_config.temp_network = ConvNetAutoencoderConfig(time_embed_dim=12,time_scale_factor=10)

    # TEMPORAL HOLLOW TRANSFORMERS
    #hidden_dim = 256
    #ctdd_config.model = BackwardRateTemporalHollowTransformerConfig()
    #ctdd_config.temp_network = TemporalHollowTransformerConfig(num_heads=2,
    #                                                           num_layers=2,
    #                                                           hidden_dim=hidden_dim,
    #                                                           ff_hidden_dim=hidden_dim*2,
    #                                                           time_embed_dim=128,
    #                                                           time_scale_factor=10)

    ctdd_config.sampler = ParametrizedSamplerConfig(num_steps=100)
    ctdd_config.trainer = CTDDTrainerConfig(device="cuda:0",
                                            num_epochs=50,
                                            save_metric_epochs=10,
                                            save_model_epochs=10,
                                            save_image_epochs=10,
                                            learning_rate=1e-3,
                                            metrics=["mse_histograms",
                                                     "histograms"])
    ctdd_trainer = CTDDTrainer(ctdd_config)
    ctdd_trainer.train_ctdd()

