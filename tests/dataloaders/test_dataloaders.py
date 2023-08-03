import torch
from graph_bridges.data.graph_dataloaders_config import EgoConfig
from graph_bridges.data.graph_dataloaders import BridgeGraphDataLoaders
from graph_bridges.configs.graphs.config_sb import SBConfig

if __name__=="__main__":
    #================================
    # upper diagonal matrix
    #================================

    device = torch.device("cpu")
    bridge_config = SBConfig(experiment_indentifier="debug")

    graph_config = EgoConfig(full_adjacency=False,flatten_adjacency=True,as_image=True)
    bridge_config.data = graph_config
    bridge_graph_dataloader = BridgeGraphDataLoaders(bridge_config,device)
    databatch = next(bridge_graph_dataloader.train().__iter__())
    adj,features = databatch[0],databatch[1]
    graph_ = bridge_graph_dataloader.transform_to_graph(adj)
    print("Expected Shape {0} Transformed Shape {1} D {2}".format(graph_config.shape_,adj.shape,graph_config.D))
    print("Back Transform Shape {0}".format(graph_.shape))

    graph_config = EgoConfig(full_adjacency=False,flatten_adjacency=True,as_image=False)
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

    graph_config = EgoConfig(full_adjacency=True,flatten_adjacency=True, as_image=True)
    bridge_config.data = graph_config
    bridge_graph_dataloader = BridgeGraphDataLoaders(bridge_config,device)
    databatch = next(bridge_graph_dataloader.train().__iter__())
    adj,features = databatch[0],databatch[1]
    graph_ = bridge_graph_dataloader.transform_to_graph(adj)
    print("Expected Shape {0} Transformed Shape {1} D {2}".format(graph_config.shape_,adj.shape,graph_config.D))
    print("Back Transform Shape {0}".format(graph_.shape))

    graph_config = EgoConfig(full_adjacency=True,flatten_adjacency=True, as_image=False)
    bridge_config.data = graph_config
    bridge_graph_dataloader = BridgeGraphDataLoaders(bridge_config,device)
    databatch = next(bridge_graph_dataloader.train().__iter__())
    adj,features = databatch[0],databatch[1]
    graph_ = bridge_graph_dataloader.transform_to_graph(adj)
    print("Expected Shape {0} Transformed Shape {1} D {2}".format(graph_config.shape_,adj.shape,graph_config.D))
    print("Back Transform Shape {0}".format(graph_.shape))

    graph_config = EgoConfig(full_adjacency=True,flatten_adjacency=False, as_image=True)
    bridge_config.data = graph_config
    bridge_graph_dataloader = BridgeGraphDataLoaders(bridge_config,device)
    databatch = next(bridge_graph_dataloader.train().__iter__())
    adj,features = databatch[0],databatch[1]
    graph_ = bridge_graph_dataloader.transform_to_graph(adj)
    print("Expected Shape {0} Transformed Shape {1} D {2}".format(graph_config.shape_,adj.shape,graph_config.D))
    print("Back Transform Shape {0}".format(graph_.shape))

    graph_config = EgoConfig(full_adjacency=True,flatten_adjacency=False, as_image=False,as_spins=False)
    bridge_config.data = graph_config
    bridge_graph_dataloader = BridgeGraphDataLoaders(bridge_config,device)
    databatch = next(bridge_graph_dataloader.train().__iter__())
    adj,features = databatch[0],databatch[1]
    graph_ = bridge_graph_dataloader.transform_to_graph(adj)
    print("Expected Shape {0} Transformed Shape {1} D {2}".format(graph_config.shape_,adj.shape,graph_config.D))
    print("Back Transform Shape {0}".format(graph_.shape))
