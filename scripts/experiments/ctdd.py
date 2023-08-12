from graph_bridges.models.trainers.ctdd_training import CTDDTrainer


if __name__=="__main__":
    from graph_bridges.configs.graphs.config_ctdd import CTDDConfig,CTDDTrainerConfig
    from graph_bridges.data.graph_dataloaders_config import EgoConfig,CommunityConfig,GridConfig
    from graph_bridges.data.graph_dataloaders_config import CommunitySmallConfig
    from graph_bridges.models.backward_rates.backward_rate_config import BackRateMLPConfig

    ctdd_config = CTDDConfig(experiment_indentifier="dario_ego",
                             experiment_name="graph",
                             experiment_type="ctdd")
    ctdd_config.data = EgoConfig(batch_size=24, full_adjacency=False)
    ctdd_config.model = BackRateMLPConfig()

    ctdd_config.trainer = CTDDTrainerConfig(device="cuda:0",
                                            num_epochs=300,
                                            save_metric_epochs=50,
                                            save_model_epochs=50,
                                            save_image_epochs=50,
                                            metrics=["graphs_plots",
                                                     #"graphs",
                                                     "histograms"])

    ctdd_trainer = CTDDTrainer(ctdd_config)
    ctdd_trainer.train_ctdd()

