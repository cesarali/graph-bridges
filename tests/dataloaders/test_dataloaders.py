import torch
from graph_bridges.data.graph_dataloaders import load_data
from graph_bridges.data.graph_dataloaders_config import EgoConfig
from graph_bridges.data.graph_dataloaders import BridgeGraphDataLoaders
from graph_bridges.configs.graphs.config_sb import BridgeConfig
if __name__=="__main__":
    #================================
    # full matrix
    #================================

    device = torch.device("cpu")
    from graph_bridges.data.graph_dataloaders import get_transforms
    from torchvision import transforms

    bridge_config = BridgeConfig(experiment_indentifier="debug")


    graph_config = EgoConfig(full_adjacency=False,flatten_adjacency=True,as_image=True)
    bridge_config.data = graph_config
    bridge_graph_dataloader = BridgeGraphDataLoaders(bridge_config,device)
    databatch = next(bridge_graph_dataloader.train().__iter__())
    adj,features = databatch[0],databatch[1]

    transform_list = get_transforms(graph_config)
    composed_transform = transforms.Compose(transform_list)
    transformed_tensor_list = composed_transform(adj)

    print(graph_config.shape)
    print(transformed_tensor_list.shape)

    graph_config = EgoConfig(full_adjacency=False,flatten_adjacency=True,as_image=False)
    bridge_config.data = graph_config
    bridge_graph_dataloader = BridgeGraphDataLoaders(bridge_config,device)
    databatch = next(bridge_graph_dataloader.train().__iter__())
    adj,features = databatch[0],databatch[1]

    transform_list = get_transforms(graph_config)
    composed_transform = transforms.Compose(transform_list)
    transformed_tensor_list = composed_transform(adj)

    print(graph_config.shape)
    print(transformed_tensor_list.shape)

    graph_config = EgoConfig(full_adjacency=True,flatten_adjacency=True, as_image=True)
    bridge_config.data = graph_config
    bridge_graph_dataloader = BridgeGraphDataLoaders(bridge_config,device)
    databatch = next(bridge_graph_dataloader.train().__iter__())
    adj,features = databatch[0],databatch[1]

    transform_list = get_transforms(graph_config)
    composed_transform = transforms.Compose(transform_list)
    transformed_tensor_list = composed_transform(adj)

    print(graph_config.shape)
    print(transformed_tensor_list.shape)

    graph_config = EgoConfig(full_adjacency=True,flatten_adjacency=True, as_image=False)
    bridge_config.data = graph_config
    bridge_graph_dataloader = BridgeGraphDataLoaders(bridge_config,device)
    databatch = next(bridge_graph_dataloader.train().__iter__())
    adj,features = databatch[0],databatch[1]

    transform_list = get_transforms(graph_config)
    composed_transform = transforms.Compose(transform_list)
    transformed_tensor_list = composed_transform(adj)

    print(graph_config.shape)
    print(transformed_tensor_list.shape)

    graph_config = EgoConfig(full_adjacency=True,flatten_adjacency=False, as_image=True)
    bridge_config.data = graph_config
    bridge_graph_dataloader = BridgeGraphDataLoaders(bridge_config,device)
    databatch = next(bridge_graph_dataloader.train().__iter__())
    adj,features = databatch[0],databatch[1]

    transform_list = get_transforms(graph_config)
    composed_transform = transforms.Compose(transform_list)
    transformed_tensor_list = composed_transform(adj)

    print(graph_config.shape)
    print(transformed_tensor_list.shape)

    graph_config = EgoConfig(full_adjacency=True,flatten_adjacency=False, as_image=False)
    bridge_config.data = graph_config
    bridge_graph_dataloader = BridgeGraphDataLoaders(bridge_config,device)
    databatch = next(bridge_graph_dataloader.train().__iter__())
    adj,features = databatch[0],databatch[1]

    transform_list = get_transforms(graph_config)
    composed_transform = transforms.Compose(transform_list)
    transformed_tensor_list = composed_transform(adj)

    print(graph_config.shape)
    print(transformed_tensor_list.shape)