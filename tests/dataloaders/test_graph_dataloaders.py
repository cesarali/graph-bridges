import os
import torch
import unittest

from graph_bridges.data.dataloaders_utils import load_dataloader
from graph_bridges.configs.graphs.graph_config_sb import SBConfig
from graph_bridges.configs.graphs.graph_config_ctdd import CTDDConfig
from graph_bridges.data.graph_dataloaders import DoucetTargetData
from graph_bridges.data.graph_dataloaders import BridgeGraphDataLoaders
from graph_bridges.data.graph_dataloaders_config import CommunityConfig

class TestDataDataloader(unittest.TestCase):
    config:CTDDConfig
    dataloader: BridgeGraphDataLoaders

    def setUp(self) -> None:
        self.config = CTDDConfig(delete=True,experiment_indentifier="testing")
        self.config.data = CommunityConfig(as_image=False,
                                           as_spins=True,
                                           batch_size=32,
                                           full_adjacency=False)
        self.device = torch.device("cpu")
        self.config.align_configurations()
        self.dataloader = load_dataloader(self.config,"data",self.device)

    def test_sample(self):
        sample_size = 40
        samples = self.dataloader.sample(sample_size=sample_size,type="train")
        self.assertTrue(samples[0].shape[0] == sample_size)
        self.assertTrue(samples[1].shape[0] == sample_size)

    def test_back_to_graph(self):
        databatch = next(self.dataloader.train().__iter__())
        x_adj_spins = databatch[0]
        x_adj = self.dataloader.transform_to_graph(x_adj_spins)
        self.assertTrue(x_adj.min() >= 0.)

    def test_databatch(self):
        databatch = next(self.dataloader.train().__iter__())
        self.assertTrue(len(databatch)==2)