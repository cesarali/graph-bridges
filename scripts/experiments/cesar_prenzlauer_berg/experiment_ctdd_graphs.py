from graph_bridges.models.trainers.ctdd_training import CTDDTrainer


if __name__=="__main__":
    from graph_bridges.configs.graphs.graph_config_ctdd import CTDDConfig
    from graph_bridges.configs.config_ctdd import CTDDTrainerConfig,ParametrizedSamplerConfig

    from graph_bridges.data.graph_dataloaders_config import EgoConfig,CommunityConfig,GridConfig,CommunitySmallConfig
    from graph_bridges.data.graph_dataloaders_config import PepperMNISTDataConfig
    from graph_bridges.data.graph_dataloaders_config import CommunitySmallConfig
    from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig
    from graph_bridges.models.backward_rates.ctdd_backward_rate_config import GaussianTargetRateImageX0PredEMAConfig
    from graph_bridges.models.networks.convnets.autoencoder import ConvNetAutoencoderConfig

    ctdd_config = CTDDConfig(experiment_indentifier="config_changes6",
                             experiment_name="graph",
                             experiment_type="ctdd")
    ctdd_config.data = EgoConfig(batch_size=24, full_adjacency=True)
    ctdd_config.model = BackRateMLPConfig()

    #ctdd_config.model = GaussianTargetRateImageX0PredEMAConfig()
    #ctdd_config.temp_network = ConvNetAutoencoderConfig(time_embed_dim=12,time_scale_factor=10)

    ctdd_config.sampler = ParametrizedSamplerConfig(num_steps=100)
    ctdd_config.trainer = CTDDTrainerConfig(device="cuda:0",
                                            num_epochs=50,
                                            save_metric_epochs=10,
                                            save_model_epochs=10,
                                            save_image_epochs=10,
                                            metrics=["graphs_plots",
                                                     "histograms"])
    ctdd_trainer = CTDDTrainer(ctdd_config)
    ctdd_trainer.train_ctdd()

