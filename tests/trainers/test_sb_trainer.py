import os
import torch
import unittest
import numpy as np
import pandas as pd
from pprint import pprint
from dataclasses import asdict

from graph_bridges.utils.test_utils import check_model_devices
from graph_bridges.data.graph_dataloaders_config import EgoConfig
from graph_bridges.models.backward_rates.backward_rate_config import BackRateMLPConfig

from graph_bridges.models.generative_models.sb import SB
from graph_bridges.configs.graphs.config_sb import TrainerConfig
from graph_bridges.models.backward_rates.backward_rate_config import BackRateMLPConfig
from graph_bridges.data.graph_dataloaders_config import EgoConfig, CommunityConfig, CommunitySmallConfig
from graph_bridges.configs.graphs.config_sb import SBConfig, ParametrizedSamplerConfig, SteinSpinEstimatorConfig
from graph_bridges.models.backward_rates.backward_rate_config import GaussianTargetRateImageX0PredEMAConfig

from graph_bridges.models.trainers.sb_training import SBTrainer
from graph_bridges.utils.test_utils import check_model_devices

class TestSBTrainer(unittest.TestCase):

    sb:SB
    sb_config:SBConfig
    sb_trainer:SBTrainer

    def setUp(self) -> None:
        self.sb_config = SBConfig(delete=True,experiment_indentifier="unittest_sb_trainer")
        self.sb_config.model = BackRateMLPConfig(time_embed_dim=14, hidden_layer=150)
        self.sb_config.stein = SteinSpinEstimatorConfig(stein_sample_size=10)
        self.sb_config.sampler = ParametrizedSamplerConfig(num_steps=5)
        self.sb_config.trainer = TrainerConfig(learning_rate=1e-3,
                                               num_epochs=6,
                                               save_metric_epochs=2,
                                               device="cuda:0",
                                               metrics=["graphs_plots","histograms"])

        self.sb_trainer = SBTrainer(self.sb_config)

    def test_training(self):
        self.sb_trainer.train_schrodinger()

    def test_sinkhorn_initialization(self):
        current_model = self.sb_trainer.sb.training_model
        past_model = self.sb_trainer.sb.reference_process
        self.sb_trainer.initialize_sinkhorn(current_model,past_model,sinkhorn_iteration=0)

    def test_metrics_login(self):
        sinkhorn_iteration = 0
        current_model = self.sb_trainer.sb.training_model
        past_model = self.sb_trainer.sb.reference_process

        self.sb_trainer.log_metrics(current_model=current_model,
                                    past_to_train_model=past_model,
                                    sinkhorn_iteration=sinkhorn_iteration,
                                    epoch=10,
                                    device=self.sb_trainer.device)

if __name__ == '__main__':
    unittest.main()
