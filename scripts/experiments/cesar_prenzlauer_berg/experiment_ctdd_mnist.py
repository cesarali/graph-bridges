from graph_bridges.models.trainers.ctdd_training import CTDDTrainer

if __name__=="__main__":

    from graph_bridges.configs.images.nist_config_ctdd import CTDDConfig
    from graph_bridges.configs.config_ctdd import CTDDTrainerConfig,ParametrizedSamplerConfig
    from graph_bridges.models.temporal_networks.convnets.autoencoder import ConvNetAutoencoderConfig
    from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig
    from graph_bridges.models.backward_rates.ctdd_backward_rate_config import GaussianTargetRateImageX0PredEMAConfig
    from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackwardRateTemporalHollowTransformerConfig
    from graph_bridges.models.temporal_networks.transformers.temporal_hollow_transformers import TemporalHollowTransformerConfig

    ctdd_config = CTDDConfig(experiment_indentifier="hollow_transformer",
                             experiment_name="mnist",
                             experiment_type="ctdd")

    #MLP
    ctdd_config.model = BackRateMLPConfig(hidden_layer=500,time_embed_dim=128)

    #CONVNET
    #ctdd_config.model = GaussianTargetRateImageX0PredEMAConfig()
    #ctdd_config.temp_network = ConvNetAutoencoderConfig()

    #TEMPORAL HOLLOW TRANSFORMERS
    ctdd_config.model = BackwardRateTemporalHollowTransformerConfig()
    ctdd_config.temp_network = TemporalHollowTransformerConfig(num_heads=1,
                                                               num_layers=1,
                                                               hidden_dim=12,
                                                               ff_hidden_dim=24)

    ctdd_config.data.batch_size = 128
    ctdd_config.data.data = "emnist"

    ctdd_config.sampler = ParametrizedSamplerConfig(num_steps=25)
    ctdd_config.trainer = CTDDTrainerConfig(device="cuda:0",
                                            num_epochs=20,
                                            save_metric_epochs=5,
                                            save_model_epochs=1,
                                            save_image_epochs=9,
                                            metrics=[])

    ctdd_trainer = CTDDTrainer(ctdd_config)
    ctdd_trainer.train_ctdd()

