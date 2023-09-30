import os
import torch
import unittest

from graph_bridges.data.dataloaders_utils import load_dataloader
from graph_bridges.configs.graphs.graph_config_sb import SBConfig
from graph_bridges.configs.graphs.graph_config_ctdd import CTDDConfig

from graph_bridges.data.graph_dataloaders import DoucetTargetData
from graph_bridges.data.graph_dataloaders import BridgeGraphDataLoaders
from graph_bridges.data.graph_dataloaders_config import CommunityConfig

from graph_bridges.models.temporal_networks.mlp.temporal_mlp import TemporalMLPConfig
from graph_bridges.models.temporal_networks.convnets.autoencoder import ConvNetAutoencoderConfig
from graph_bridges.models.temporal_networks.transformers.temporal_hollow_transformers import TemporalHollowTransformerConfig

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
        self.dataloader_data = load_dataloader(self.config,"data",self.device)
        self.dataloader_target = load_dataloader(self.config,"target",self.device)

    def test_sample(self):
        sample_size = 40
        samples = self.dataloader_data.sample(sample_size=sample_size,type="train")
        self.assertTrue(samples[0].shape[0] == sample_size)
        self.assertTrue(samples[1].shape[0] == sample_size)

    def test_back_to_graph(self):
        databatch = next(self.dataloader_data.train().__iter__())
        x_adj_spins = databatch[0]
        x_adj = self.dataloader_data.transform_to_graph(x_adj_spins)
        self.assertTrue(x_adj.min() >= 0.)

    def test_databatch(self):
        databatch = next(self.dataloader_data.train().__iter__())
        self.assertTrue(len(databatch)==2)

    def test_sizes(self):
        number_of_steps_data = 0
        for databatch in self.dataloader_data.train():
            number_of_steps_data += databatch[0].shape[0]
        self.assertTrue(number_of_steps_data,self.config.data.training_size)

        number_of_steps_target = 0
        for databatch in self.dataloader_target.train():
            number_of_steps_target += databatch[0].shape[0]
        self.assertTrue(number_of_steps_target,self.config.data.training_size)


    def test_shapes(self):
        from graph_bridges.configs.graphs.graph_config_sb import SBConfig
        from graph_bridges.data.dataloaders_utils import load_dataloader, check_sizes

        self.batch_size = 12
        self.num_time_steps = 8

        # ================
        sb_config = SBConfig(experiment_indentifier="sb_unittest")
        sb_config.data = CommunityConfig(as_image=False, batch_size=self.batch_size, full_adjacency=True)
        sb_config.temp_network = ConvNetAutoencoderConfig()
        data_sizes = check_sizes(sb_config)

        print(f"Shape {data_sizes.temporal_net_expected_shape}")
        print(f"Data Size {data_sizes.training_size}")


        # ================
        sb_config = SBConfig(experiment_indentifier="sb_unittest")
        sb_config.data = CommunityConfig(as_image=False, batch_size=self.batch_size, full_adjacency=True)
        sb_config.temp_network = TemporalHollowTransformerConfig()
        data_sizes = check_sizes(sb_config)

        print(f"Shape {data_sizes.temporal_net_expected_shape}")
        print(f"Data Size {data_sizes.training_size}")