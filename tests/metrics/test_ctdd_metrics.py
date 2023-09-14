import os
import torch
import unittest
import numpy as np
import pandas as pd
import networkx as nx
from pprint import pprint
from dataclasses import asdict

from graph_bridges.models.generative_models.ctdd import CTDD
from graph_bridges.utils.test_utils import check_model_devices
from graph_bridges.configs.graphs.graph_config_ctdd import CTDDConfig
from graph_bridges.data.graph_dataloaders_config import EgoConfig

from graph_bridges.models.metrics.ctdd_metrics import graph_metrics_for_ctdd,marginal_histograms_for_ctdd
from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig


class TestCTDDMetrics(unittest.TestCase):

    ctdd_config: CTDDConfig
    ctdd: CTDD

    def setUp(self) -> None:
        self.ctdd_config = CTDDConfig(experiment_indentifier="ctdd_unittest",delete=True)
        self.ctdd_config.data = EgoConfig(as_image=False, batch_size=32, full_adjacency=False)
        self.ctdd_config.model = BackRateMLPConfig()
        self.ctdd_config.initialize_new_experiment()

        self.device = torch.device("cuda:0")
        self.ctdd = CTDD()
        self.ctdd.create_new_from_config(self.ctdd_config, self.device)

    def test_graph_generation(self):
        number_of_graph_to_generate = 12
        graph_list = self.ctdd.generate_graphs(number_of_graph_to_generate)
        self.assertTrue(len(graph_list) == number_of_graph_to_generate)
        self.assertIsInstance(graph_list[0],nx.Graph)


if __name__ == '__main__':
    unittest.main()
