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
from graph_bridges.configs.graphs.config_ctdd import CTDDConfig
from graph_bridges.data.graph_dataloaders_config import EgoConfig
from graph_bridges.models.backward_rates.backward_rate_config import GaussianTargetRateImageX0PredEMAConfig


class TestCTDD(unittest.TestCase):

    ctdd_config: CTDDConfig
    ctdd: CTDD

    def setUp(self) -> None:
        self.ctdd_config = CTDDConfig(experiment_indentifier="ctdd_unittest",delete=True)
        self.ctdd_config.data = EgoConfig(as_image=False, batch_size=32, full_adjacency=False)
        self.ctdd_config.model = GaussianTargetRateImageX0PredEMAConfig(time_embed_dim=32, fix_logistic=False)
        self.ctdd_config.initialize_new_experiment()

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.ctdd = CTDD()
        self.ctdd.create_new_from_config(self.ctdd_config, self.device)

    @unittest.skipIf(not torch.cuda.is_available(),"Cuda Not Available")
    def test_gpu(self):
        print("Test GPU")
        self.assertTrue(self.device == check_model_devices(self.ctdd.model))

    def test_pipeline(self):
        print("Test Pipeline")
        x = self.ctdd.pipeline(self.ctdd.model, 36)
        self.assertIsInstance(x,torch.Tensor)

    def test_graph_generation(self):
        number_of_graph_to_generate = 12
        graph_list = self.ctdd.generate_graphs(number_of_graph_to_generate)
        self.assertTrue(len(graph_list) == number_of_graph_to_generate)
        self.assertIsInstance(graph_list[0],nx.Graph)

@unittest.skip
class TestCTDDCifar10(unittest.TestCase):

    ctdd_config: CTDDConfig
    ctdd: CTDD

    def setUp(self) -> None:
        from graph_bridges.models.backward_rates.backward_rate_config import GaussianTargetRateImageX0PredEMAConfig
        from graph_bridges.data.image_dataloader_config import DiscreteCIFAR10Config
        from graph_bridges.configs.images.cifar10_config_ctdd import CTDDConfig
        from graph_bridges.models.generative_models.ctdd import CTDD

        config = CTDDConfig()
        config.data = DiscreteCIFAR10Config(batch_size=28)
        config.trainer.device = "cpu"

        # device
        device = torch.device(config.trainer.device)

        # model
        self.ctdd = CTDD()
        self.ctdd.create_new_from_config(config, device)

    def test_pipeline(self):
        x = self.ctdd.pipeline(self.ctdd.model, 36)
        print(x)

if __name__ == '__main__':
    unittest.main()
