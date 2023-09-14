from graph_bridges.models.trainers.ctdd_training import CTDDTrainer

if __name__=="__main__":

    from graph_bridges.configs.graphs.config_ctdd import CTDDConfig, CTDDTrainerConfig, ParametrizedSamplerConfig
    from graph_bridges.data.graph_dataloaders_config import EgoConfig,CommunityConfig,GridConfig
    from graph_bridges.data.graph_dataloaders_config import CommunitySmallConfig
    from graph_bridges.models.backward_rates.backward_rate_config import BackRateMLPConfig, GaussianTargetRateImageX0PredEMAConfig

    ctdd_config = CTDDConfig(experiment_indentifier="dario_test_UpperDiag_hid300",
                             experiment_name="graph",
                             experiment_type="ctdd", delete=True)
    
    ctdd_config.data = CommunitySmallConfig(batch_size=32, full_adjacency=False)
    ctdd_config.sampler = ParametrizedSamplerConfig(num_steps=100)
    ctdd_config.model = BackRateMLPConfig(time_embed_dim=24, hidden_layer=300)

    ctdd_config.trainer = CTDDTrainerConfig(device="cuda:0",
                                            num_epochs=1000,
                                            save_metric_epochs=250,
                                            save_model_epochs=100,
                                            save_image_epochs=100,
                                            metrics=["graphs_plots",
                                                     "graphs",
                                                     "histograms"])

    ctdd_trainer = CTDDTrainer(ctdd_config)
    ctdd_trainer.train_ctdd()

