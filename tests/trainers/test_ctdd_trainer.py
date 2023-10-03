import os
import torch
import unittest
import numpy as np
import pandas as pd
from pprint import pprint
from dataclasses import asdict

from graph_bridges.models.generative_models.ctdd import CTDD
from graph_bridges.utils.test_utils import check_model_devices
from graph_bridges.configs.graphs.graph_config_ctdd import CTDDConfig
from graph_bridges.data.graph_dataloaders_config import EgoConfig
from graph_bridges.data.graph_dataloaders_config import CommunityConfig
from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig

from graph_bridges.utils.test_utils import check_model_devices
from graph_bridges.configs.config_ctdd import CTDDTrainerConfig
from graph_bridges.models.trainers.ctdd_training import CTDDTrainer


class TestCTDDTrainer(unittest.TestCase):

    ctdd_config: CTDDConfig
    ctdd: CTDD

    def setUp(self) -> None:
        self.ctdd_config = CTDDConfig(experiment_indentifier="ctdd_trainer_unittest",delete=True)
        self.ctdd_config.data = EgoConfig(batch_size=32, full_adjacency=False)
        self.ctdd_config.model = BackRateMLPConfig()
        self.ctdd_config.trainer = CTDDTrainerConfig(learning_rate=1e-3,
                                                     num_epochs=2,
                                                     save_metric_epochs=2,
                                                     device="cuda:0",
                                                     metrics=["graphs_plots",
                                                        "histograms"])
        self.ctdd_trainer = CTDDTrainer(self.ctdd_config)

    def test_trained(self):
        print("Test Initialization")
        original_determinism = torch.use_deterministic_algorithms(False)
        self.ctdd_trainer.train_ctdd()

if __name__ == '__main__':
    unittest.main()