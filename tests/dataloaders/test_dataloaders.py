import unittest
import torch

from graph_bridges.models.generative_models.sb import SB
from graph_bridges.configs.graphs.config_sb import SBConfig
from graph_bridges.data.dataloaders import DoucetTargetData
from graph_bridges.data.dataloaders_utils import load_dataloader
from graph_bridges.data.graph_dataloaders import BridgeGraphDataLoaders

from graph_bridges.data.graph_dataloaders_config import EgoConfig,CommunityConfig


class TestDataDataloader(unittest.TestCase):
    config:SBConfig
    dataloader: BridgeGraphDataLoaders

    def setUp(self) -> None:
        self.config = SBConfig(delete=True,experiment_indentifier="testing")
        self.config.data = CommunityConfig(as_image=False, batch_size=32, full_adjacency=False)
        self.device = torch.device("cpu")
        self.config.align_configurations()
        self.dataloader = load_dataloader(self.config,"data",self.device)

    def test_sample(self):
        sample_size = 40
        samples = self.dataloader.sample(sample_size=sample_size,type="train")
        self.assertTrue(samples[0].shape[0] == sample_size)
        self.assertTrue(samples[1].shape[0] == sample_size)

    def test_databatch(self):
        databatch = next(self.dataloader.train().__iter__())
        self.assertTrue(len(databatch)==2)

class TestTargetDataloader(unittest.TestCase):
    config:SBConfig
    dataloader:DoucetTargetData

    def setUp(self) -> None:
        self.config = SBConfig(delete=True,experiment_indentifier="testing")
        self.config.data = CommunityConfig(as_image=False, batch_size=32, full_adjacency=False)
        device = torch.device("cpu")

        self.config.align_configurations()
        self.dataloader = load_dataloader(self.config,"target",device)

    def test_sample(self):
        sample_size = 40
        databath = self.dataloader.sample(sample_size)
        self.assertTrue(databath[0].shape[0]==sample_size)


if __name__=="__main__":
    unittest.main()

    """
    #================================
    # upper diagonal matrix
    #================================
    GeneralConfig = MNISTDataConfig

    device = torch.device("cpu")
    bridge_config = SBConfig(experiment_indentifier="debug")

    graph_config = GeneralConfig(full_adjacency=False,flatten_adjacency=True,as_image=True)
    bridge_config.data = graph_config
    bridge_graph_dataloader = BridgeGraphDataLoaders(bridge_config,device)
    databatch = next(bridge_graph_dataloader.train().__iter__())
    adj,features = databatch[0],databatch[1]
    graph_ = bridge_graph_dataloader.transform_to_graph(adj)
    print("Expected Shape {0} Transformed Shape {1} D {2}".format(graph_config.shape_,adj.shape,graph_config.D))
    print("Back Transform Shape {0}".format(graph_.shape))

    graph_config = GeneralConfig(full_adjacency=False,flatten_adjacency=True,as_image=False)
    bridge_config.data = graph_config
    bridge_graph_dataloader = BridgeGraphDataLoaders(bridge_config,device)
    databatch = next(bridge_graph_dataloader.train().__iter__())
    adj,features = databatch[0],databatch[1]
    graph_ = bridge_graph_dataloader.transform_to_graph(adj)
    print("Expected Shape {0} Transformed Shape {1} D {2}".format(graph_config.shape_,adj.shape,graph_config.D))
    print("Back Transform Shape {0}".format(graph_.shape))

    #================================
    # full matrix
    #================================

    graph_config = GeneralConfig(full_adjacency=True,flatten_adjacency=True, as_image=True)
    bridge_config.data = graph_config
    bridge_graph_dataloader = BridgeGraphDataLoaders(bridge_config,device)
    databatch = next(bridge_graph_dataloader.train().__iter__())
    adj,features = databatch[0],databatch[1]
    graph_ = bridge_graph_dataloader.transform_to_graph(adj)
    print("Expected Shape {0} Transformed Shape {1} D {2}".format(graph_config.shape_,adj.shape,graph_config.D))
    print("Back Transform Shape {0}".format(graph_.shape))

    graph_config = GeneralConfig(full_adjacency=True,flatten_adjacency=True, as_image=False)
    bridge_config.data = graph_config
    bridge_graph_dataloader = BridgeGraphDataLoaders(bridge_config,device)
    databatch = next(bridge_graph_dataloader.train().__iter__())
    adj,features = databatch[0],databatch[1]
    graph_ = bridge_graph_dataloader.transform_to_graph(adj)
    print("Expected Shape {0} Transformed Shape {1} D {2}".format(graph_config.shape_,adj.shape,graph_config.D))
    print("Back Transform Shape {0}".format(graph_.shape))

    graph_config = GeneralConfig(full_adjacency=True,flatten_adjacency=False, as_image=True)
    bridge_config.data = graph_config
    bridge_graph_dataloader = BridgeGraphDataLoaders(bridge_config,device)
    databatch = next(bridge_graph_dataloader.train().__iter__())
    adj,features = databatch[0],databatch[1]
    graph_ = bridge_graph_dataloader.transform_to_graph(adj)
    print("Expected Shape {0} Transformed Shape {1} D {2}".format(graph_config.shape_,adj.shape,graph_config.D))
    print("Back Transform Shape {0}".format(graph_.shape))

    graph_config = GeneralConfig(full_adjacency=True,flatten_adjacency=False, as_image=False,as_spins=False)
    bridge_config.data = graph_config
    bridge_graph_dataloader = BridgeGraphDataLoaders(bridge_config,device)
    databatch = next(bridge_graph_dataloader.train().__iter__())
    adj,features = databatch[0],databatch[1]
    graph_ = bridge_graph_dataloader.transform_to_graph(adj)
    print("Expected Shape {0} Transformed Shape {1} D {2}".format(graph_config.shape_,adj.shape,graph_config.D))
    print("Back Transform Shape {0}".format(graph_.shape))
    """