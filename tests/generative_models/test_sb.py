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
from graph_bridges.configs.graphs.graph_config_sb import SBConfig
from graph_bridges.configs.config_sb import  ParametrizedSamplerConfig, SteinSpinEstimatorConfig


class TestSB(unittest.TestCase):
    """
    Test the Schrödinger Bridge Generative Model with a super basic MLP as
    backward rate model
    """
    sb_config: SBConfig
    sb: SB

    def setUp(self) -> None:
        self.sb_config = SBConfig(experiment_indentifier="sb_unittest")
        self.sb_config.data = EgoConfig(as_image=False, batch_size=5, full_adjacency=False)
        self.sb_config.flip_estimator = SteinSpinEstimatorConfig(stein_sample_size=5)
        self.sb_config.sampler = ParametrizedSamplerConfig(num_steps=5)

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.sb = SB()
        self.sb.create_new_from_config(self.sb_config, self.device)

    def test_gpu(self):
        print("Test GPU")
        self.assertTrue(self.device == check_model_devices(self.sb.training_model))
        self.assertTrue(self.device == check_model_devices(self.sb.past_model))

    def test_pipeline(self):
        print("Test Pipeline")
        x_end, times = self.sb.pipeline(None, 0, self.device, return_path=True)
        self.assertIsInstance(x_end,torch.Tensor)
        self.assertIsInstance(times,torch.Tensor)

    def test_graph_generation(self):
        number_of_graph_to_generate = 12
        target_graph_list = self.sb.generate_graphs(number_of_graph_to_generate,self.sb.training_model,sinkhorn_iteration=1)
        self.assertTrue(len(target_graph_list) == number_of_graph_to_generate)
        self.assertIsInstance(target_graph_list[0],nx.Graph)

        data_graph_list = self.sb.generate_graphs(number_of_graph_to_generate,self.sb.reference_process,sinkhorn_iteration=0)
        self.assertTrue(len(data_graph_list) == number_of_graph_to_generate)
        self.assertIsInstance(data_graph_list[0],nx.Graph)


if __name__ == '__main__':
    unittest.main()

