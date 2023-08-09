import torch

from graph_bridges.models.generative_models.sb import SB
from graph_bridges.configs.graphs.config_sb import SBConfig
from graph_bridges.data.dataloaders_utils import load_dataloader
from graph_bridges.data.graph_dataloaders_config import EgoConfig,CommunityConfig
from graph_bridges.data.graph_dataloaders import BridgeGraphDataLoaders


if __name__=="__main__":

    config = SBConfig(delete=True,experiment_indentifier="testing")
    config.data = CommunityConfig(as_image=False, batch_size=32, full_adjacency=False)
    device = torch.device("cpu")
    sb = SB(config, device)

    config.align_configurations()
    data_dataloader = load_dataloader(config,"data",device)
    target_dataloader = load_dataloader(config,"target",device)

    batch_number = 0
    for a in target_dataloader.train():
        batch_number += 1
    print(a[0].shape)
    print(batch_number)

    batch_number = 0
    for a in data_dataloader.train():
        batch_number += 1
    print(a[0].shape)
    print(batch_number)

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