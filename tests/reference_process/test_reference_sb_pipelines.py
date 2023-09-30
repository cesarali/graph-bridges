import os
import torch
import unittest
import numpy as np
import pandas as pd
import networkx as nx
from pprint import pprint
from dataclasses import asdict

from graph_bridges.models.generative_models.sb import SB
from graph_bridges.utils.test_utils import check_model_devices
from graph_bridges.data.graph_dataloaders_config import EgoConfig

from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig
from graph_bridges.configs.config_sb import ParametrizedSamplerConfig, SteinSpinEstimatorConfig

class TestSB(unittest.TestCase):
    """
    Test the Schrödinger Bridge Generative Model with a super basic MLP as

    backward rate model
    """
    from graph_bridges.configs.spin_glass.spin_glass_config_sb import SBConfig

    sb_config: SBConfig
    sb: SB

    def setUp(self) -> None:
        from graph_bridges.configs.spin_glass.spin_glass_config_sb import SBConfig
        from graph_bridges.models.temporal_networks.mlp.temporal_mlp import TemporalMLPConfig

        self.sb_config = SBConfig(experiment_indentifier="sb_unittest")

        self.sb_config.model = BackRateMLPConfig(time_embed_dim=12)
        self.sb_config.temp_network = TemporalMLPConfig()

        self.sb_config.flip_estimator = SteinSpinEstimatorConfig(stein_sample_size=5)
        self.sb_config.sampler = ParametrizedSamplerConfig(num_steps=5, step_type="TauLeaping")

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else self.device = torch.device("cpu")

        self.sb = SB()
        self.sb.create_new_from_config(self.sb_config, self.device)

        databatch = next(self.sb.data_dataloader.train().__iter__())
        self.x_ajd = databatch[0].to(self.device)
        self.x_features = databatch[1].to(self.device)

    #=============================================
    # WITH REFERENCE PROCESS
    #=============================================
    def test_pipeline_reference_no_path(self):
        sample_size = 50
        x_end = self.sb.pipeline(None, 0, self.device, sample_size=sample_size, return_path=False)
        self.assertTrue(x_end.shape[0] == sample_size)
        self.assertIsInstance(x_end,torch.Tensor)

    def test_pipeline_reference_with_path(self):
        x_end, times = self.sb.pipeline(None, 0, self.device, return_path=True)
        self.assertIsInstance(x_end,torch.Tensor)
        self.assertIsInstance(times,torch.Tensor)
        self.assertTrue(len(x_end.shape) == 2)
        self.assertTrue(len(times.shape) == 1)

    def test_pipeline_reference_with_pathshape(self):
        x_end, times = self.sb.pipeline(None, 0, self.device, return_path=True,return_path_shape=True)
        self.assertIsInstance(x_end,torch.Tensor)
        self.assertIsInstance(times,torch.Tensor)
        self.assertTrue(len(x_end.shape) == 3)
        self.assertTrue(len(times.shape) == 2)

    def test_pipeline_reference_with_path_with_start(self):
        x_end, times = self.sb.pipeline(None, 0, self.device, self.x_ajd, return_path=True)
        self.assertIsInstance(x_end,torch.Tensor)
        self.assertIsInstance(times,torch.Tensor)

    def test_pipeline_reference_with_path_with_start(self):
        x_end, times = self.sb.pipeline(None, 0, self.device, self.x_ajd, return_path=True)
        print(x_end.shape)
        print(times.shape)

if __name__ == '__main__':
    unittest.main()