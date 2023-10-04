
from graph_bridges.models.trainers.ctdd_training import CTDDTrainer

if __name__=="__main__":
    from graph_bridges.configs.spin_glass.spin_glass_config_ctdd import CTDDConfig
    from graph_bridges.configs.config_ctdd import CTDDTrainerConfig,ParametrizedSamplerConfig
    from graph_bridges.data.spin_glass_dataloaders_config import ParametrizedSpinGlassHamiltonianConfig

    from graph_bridges.data.graph_dataloaders_config import CommunitySmallConfig
    from graph_bridges.data.graph_dataloaders_config import EgoConfig,CommunityConfig,GridConfig,CommunitySmallConfig

    from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig
    from graph_bridges.models.temporal_networks.convnets.autoencoder import ConvNetAutoencoderConfig
    from graph_bridges.models.backward_rates.ctdd_backward_rate_config import GaussianTargetRateImageX0PredEMAConfig
    from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackwardRateTemporalHollowTransformerConfig
    from graph_bridges.models.temporal_networks.transformers.temporal_hollow_transformers import TemporalHollowTransformerConfig
    from graph_bridges.models.temporal_networks.mlp.temporal_mlp import TemporalMLPConfig

    ctdd_config = CTDDConfig(experiment_indentifier="mlp",
                             experiment_name="spin_glass",
                             experiment_type="ctdd")
    
    ctdd_config.data = ParametrizedSpinGlassHamiltonianConfig(data="bernoulli_spins_0.2",
                                                              bernoulli_spins=True,
                                                              bernoulli_probability=0.2,
                                                              number_of_spins=50,
                                                              delete_data=True,
                                                              batch_size=32)

    # MLP
    ctdd_config.model = BackRateMLPConfig()
    ctdd_config.temp_network = TemporalMLPConfig(hidden_dim=100,time_embed_dim=50)

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

    num_epochs = 50
    ctdd_config.sampler = ParametrizedSamplerConfig(num_steps=100)
    ctdd_config.trainer = CTDDTrainerConfig(device="cuda:0",
                                            num_epochs=50,
                                            save_metric_epochs=int(.25*num_epochs),
                                            save_model_epochs=int(.25*num_epochs),
                                            save_image_epochs=int(.25*num_epochs),
                                            learning_rate=1e-3,
                                            metrics=["histograms"])
    ctdd_trainer = CTDDTrainer(ctdd_config)
    ctdd_trainer.train_ctdd()

