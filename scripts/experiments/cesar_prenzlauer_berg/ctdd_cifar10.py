from graph_bridges.models.trainers.ctdd_training import CTDDTrainer


if __name__=="__main__":
    from graph_bridges.configs.images.cifar10_config_ctdd import CTDDConfig,CTDDTrainerConfig,ParametrizedSamplerConfig

    from graph_bridges.data.graph_dataloaders_config import CommunitySmallConfig
    from graph_bridges.models.backward_rates.backward_rate_config import GaussianTargetRateImageX0PredEMAConfig

    ctdd_config = CTDDConfig(experiment_indentifier=None,
                             experiment_name="cifar10",
                             experiment_type="ctdd",)

    ctdd_config.data.batch_size = 12
    ctdd_config.trainer = CTDDTrainerConfig(device="cuda:0",
                                            num_epochs=10,
                                            save_metric_epochs=10,
                                            save_model_epochs=10,
                                            save_image_epochs=10,
                                            metrics=[])

    ctdd_trainer = CTDDTrainer(ctdd_config)
    ctdd_trainer.train_ctdd()

