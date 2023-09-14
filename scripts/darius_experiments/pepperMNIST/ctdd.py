from graph_bridges.models.trainers.ctdd_training import CTDDTrainer

if __name__=="__main__":

    from graph_bridges.configs.graphs.config_ctdd import CTDDConfig, CTDDTrainerConfig, ParametrizedSamplerConfig
    from graph_bridges.data.graph_dataloaders_config import PepperMNISTDataConfig, PepperCIFARDataConfig
    from graph_bridges.models.backward_rates.backward_rate_config import BackRateMLPConfig, GaussianTargetRateImageX0PredEMAConfig

    ctdd_config = CTDDConfig(experiment_indentifier=None,
                             experiment_name="pepperMNIST",
                             experiment_type="ctdd")
    
    ctdd_config.data = PepperMNISTDataConfig(batch_size=512, full_adjacency=True)
    ctdd_config.sampler = ParametrizedSamplerConfig(num_steps=25)
    ctdd_config.model = GaussianTargetRateImageX0PredEMAConfig(time_embed_dim=24)

    ctdd_config.trainer = CTDDTrainerConfig(device="cuda:0",
                                            num_epochs=10,
                                            save_metric_epochs=2,
                                            save_model_epochs=2,
                                            save_image_epochs=2,
                                            metrics=["histograms"])

    ctdd_trainer = CTDDTrainer(ctdd_config)
    ctdd_trainer.train_ctdd()

