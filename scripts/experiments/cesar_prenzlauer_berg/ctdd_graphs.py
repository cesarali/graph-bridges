from graph_bridges.models.trainers.ctdd_training import CTDDTrainer


if __name__=="__main__":
    from graph_bridges.configs.graphs.config_ctdd import CTDDConfig,CTDDTrainerConfig,ParametrizedSamplerConfig
    from graph_bridges.data.graph_dataloaders_config import EgoConfig,CommunityConfig,GridConfig,CommunitySmallConfig
    from graph_bridges.data.graph_dataloaders_config import PepperMNISTDataConfig
    from graph_bridges.data.graph_dataloaders_config import CommunitySmallConfig
    from graph_bridges.models.backward_rates.backward_rate_config import BackRateMLPConfig
    from graph_bridges.models.backward_rates.backward_rate_config import GaussianTargetRateImageX0PredEMAConfig

    ctdd_config = CTDDConfig(experiment_indentifier=None,
                             experiment_name="graph",
                             experiment_type="ctdd")
    ctdd_config.data = CommunitySmallConfig(batch_size=24, full_adjacency=False)

    ctdd_config.model = BackRateMLPConfig()
    #ctdd_config.model = GaussianTargetRateImageX0PredEMAConfig(time_embed_dim=12)
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

