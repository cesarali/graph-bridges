from graph_bridges.models.trainers.sb_training import SBTrainer

if __name__=="__main__":
    # CONFIGURATIONS IMPORT
    from graph_bridges.configs.graphs.graph_config_sb import SBConfig, SBTrainerConfig
    from graph_bridges.configs.graphs.graph_config_sb import SteinSpinEstimatorConfig
    from graph_bridges.configs.graphs.graph_config_sb import ParametrizedSamplerConfig
    from graph_bridges.configs.graphs.graph_config_sb import get_sb_config_from_file

    # DATA CONFIGS
    from graph_bridges.data.graph_dataloaders_config import CommunitySmallConfig, EgoConfig
    # BACKWARD RATES CONFIGS
    from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig, \
        GaussianTargetRateImageX0PredEMAConfig

    # ===========================================
    # MODEL SET UP
    # ===========================================
    sb_config = SBConfig(delete=True,
                         experiment_name="graph",
                         experiment_type="sb",
                         experiment_indentifier=None)

    sb_config.data = EgoConfig(batch_size=10, full_adjacency=False)
    sb_config.flip_estimator = SteinSpinEstimatorConfig(stein_sample_size=200,
                                                        stein_epsilon=0.23)
    sb_config.sampler = ParametrizedSamplerConfig(num_steps=10)
    sb_config.trainer = SBTrainerConfig(learning_rate=1e-2,
                                        num_epochs=3000,
                                        save_metric_epochs=300,
                                        save_model_epochs=300,
                                        save_image_epochs=300,
                                        device="cuda:0",
                                        metrics=["graphs_plots",
                                                 "histograms"])
    # ========================================
    # TRAIN
    # ========================================
    sb_trainer = SBTrainer(sb_config)
    sb_trainer.train_schrodinger()