import os
import torch
import pickle
import numpy as np
import networkx as nx
from typing import List,Dict,Tuple,Union
from torch.utils.data import TensorDataset, DataLoader
from graph_bridges.configs.graphs.config_sb import BridgeConfig
from graph_bridges.utils.graph_utils import init_features, graphs_to_tensor
from torchtyping import TensorType
from graph_bridges.data.graph_dataloaders_config import CommunityConfig, GraphDataConfig
import torchvision.transforms as transforms

def from_networkx_to_spins(graph_,upper_diagonal_indices,full_adjacency=False):
    adjacency_ = nx.to_numpy_array(graph_)
    if full_adjacency:
        spins = (-1.) ** (adjacency_.flatten() + 1)
    else:
        just_upper_edges = adjacency_[upper_diagonal_indices]
        spins = (-1.) ** (just_upper_edges.flatten() + 1)
    return spins


# Create a custom transformation class
class UpperDiagonalIndicesTransform:

    def __call__(self, tensor):
        if  len(tensor.shape) == 3:
            batch_size = tensor.shape[0]
            # Get the upper diagonal entries without zero-padding with the batch as the first dimension
            upper_diagonal_entries = tensor.masked_select(torch.triu(torch.ones_like(tensor), diagonal=1).bool())
            upper_diagonal_entries = upper_diagonal_entries.reshape(batch_size, -1)
            return upper_diagonal_entries
        else:
            raise Exception("Wrong Tensor Shape in Transform")

class UnsqueezeTensorTransform:

    def __init__(self,axis=0):
        self.axis = axis
    def __call__(self, tensor:torch.Tensor):
        return tensor.unsqueeze(self.axis)

class BinaryTensorToSpinsTransform:

    def __call__(self, binary_tensor):
        spins = (-1.) ** (binary_tensor + 1)
        return spins

class SpinsToBinaryTensor:

    def __call__(self, spins):
        binary_tensor = int( (1. + spins)*.5)
        return binary_tensor

def get_transforms(config:GraphDataConfig):
    flatten_transform = transforms.Lambda(lambda x: x.reshape(x.shape[0], -1))
    if config.flatten_adjacency:
        if config.full_adjacency:
            if config.as_image:
                transform_list = [flatten_transform,UnsqueezeTensorTransform(1),UnsqueezeTensorTransform(1)]
            else:
                transform_list = [flatten_transform]
        else:
            if config.as_image:
                transform_list = [UpperDiagonalIndicesTransform(),UnsqueezeTensorTransform(1),UnsqueezeTensorTransform(1)]
            else:
                transform_list = [UpperDiagonalIndicesTransform()]
    else:
        if config.full_adjacency:
            if config.as_image:
                transform_list = [UnsqueezeTensorTransform(1)]
            else:
                transform_list = []
        else:  # no flatten no full adjacency
            raise Exception("No Flatten and No Full Adjacency incompatible for data")

    return transform_list


class BridgeGraphDataLoaders:
    """

    """
    graph_data_config : GraphDataConfig

    def __init__(self,config:BridgeConfig,device):
        """

        :param config:
        :param device:
        """
        #config.max_node_num
        self.graph_data_config = config.data
        self.device = device

        train_graph_list, test_graph_list = self.read_graph_lists()

        train_adjs_tensor,train_x_tensor = self.graph_to_tensor_and_features(train_graph_list,
                                                                             self.graph_data_config.init,
                                                                             self.graph_data_config.max_node_num,
                                                                             self.graph_data_config.max_feat_num)
        self.train_dataloader_ = self.create_dataloaders(train_adjs_tensor,train_x_tensor)

        test_adjs_tensor,test_x_tensor = self.graph_to_tensor_and_features(test_graph_list,
                                                                           self.graph_data_config.init,
                                                                           self.graph_data_config.max_node_num,
                                                                           self.graph_data_config.max_feat_num)

        self.test_dataloader_ = self.create_dataloaders(test_adjs_tensor,test_x_tensor)

    def train(self):
        return self.train_dataloader_

    def test(self):
        return self.test_dataloader_

    def upper_indices(self,cfg):
        if not self.full_adjacency:
            self.number_of_spins = np.triu_indices(self.number_of_nodes, k=1)[0].shape[0]
        else:
            self.number_of_spins = self.number_of_nodes ** 2
        cfg.number_of_spins = self.number_of_spins

        self.upper_diagonal_indices = np.triu_indices(self.number_of_nodes, k=1)

    def create_dataloaders(self,x_tensor, adjs_tensor):
        train_ds = TensorDataset(x_tensor, adjs_tensor)
        train_dl = DataLoader(train_ds,
                              batch_size=self.graph_data_config.batch_size,
                              shuffle=True)
        return train_dl

    def graph_to_tensor_and_features(self,
                                     graph_list:List[nx.Graph],
                                     init:str="zeros",
                                     max_node_num:int=None,
                                     max_feat_num:int=10)->(TensorType["number_of_graphs","max_node_num","max_node_num"],
                                                            TensorType["number_of_graphs","max_feat_num"]):
        """

        :return:adjs_tensor,x_tensor
        """
        if max_node_num is None:
            max_node_num = max([g.number_of_nodes() for g in graph_list])
        adjs_tensor = graphs_to_tensor(graph_list,max_node_num)
        x_tensor = init_features(init,adjs_tensor,max_feat_num)
        return adjs_tensor,x_tensor

    def read_graph_lists(self)->Tuple[List[nx.Graph]]:
        """

        :return: train_graph_list, test_graph_list

        """
        data_dir = self.graph_data_config.dir
        file_name = self.graph_data_config.data
        file_path = os.path.join(data_dir, file_name)
        with open(file_path + '.pkl', 'rb') as f:
            graph_list = pickle.load(f)
        test_size = int(self.graph_data_config.test_split * len(graph_list))
        train_graph_list, test_graph_list = graph_list[test_size:], graph_list[:test_size]

        return train_graph_list, test_graph_list


def load_dataset(data_dir='data', file_name=None, need_set=False):
    file_path = os.path.join(data_dir, file_name)
    with open(file_path + '.pkl', 'rb') as f:
        graph_list = pickle.load(f)
    return graph_list

def graphs_to_dataloader(graph_data_config, graph_list):
    adjs_tensor = graphs_to_tensor(graph_list, graph_data_config.max_node_num)
    x_tensor = init_features(graph_data_config.init, adjs_tensor, graph_data_config.max_feat_num)

    train_ds = TensorDataset(x_tensor, adjs_tensor)
    train_dl = DataLoader(train_ds, batch_size=graph_data_config.batch_size, shuffle=True)
    return train_dl

def dataloader(graph_data_config, get_graph_list=False):
    graph_list = load_dataset(data_dir=graph_data_config.dir, file_name=graph_data_config.data)
    test_size = int(graph_data_config.test_split * len(graph_list))
    train_graph_list, test_graph_list = graph_list[test_size:], graph_list[:test_size]
    if get_graph_list:
        return train_graph_list, test_graph_list

    return graphs_to_dataloader(graph_data_config, train_graph_list), graphs_to_dataloader(graph_data_config, test_graph_list)

def load_data(graph_data_config, get_graph_list=False):
    if graph_data_config.data in ['QM9', 'ZINC250k']:
        from graph_bridges.data.graph_dataloaders_mol import  dataloader as mol_dataloader
        return mol_dataloader(graph_data_config, get_graph_list)
    else:
        return dataloader(graph_data_config, get_graph_list)


if __name__=="__main__":
    from dataclasses import asdict

    data_config = CommunityConfig()
    train_loader, test_loader = load_data(data_config)
    databatch = next(train_loader.__iter__())
    device = torch.device("cpu")

    """
    """

    bridge_config = BridgeConfig(experiment_indentifier="debug")
    data_config = CommunityConfig()
    bridge_config.data = data_config
    bridge_graph_dataloader = BridgeGraphDataLoaders(bridge_config,device)

    databatch = next(bridge_graph_dataloader.train().__iter__())
    adj = databatch[0]
    features = databatch[1]


    transform_list = get_transforms(data_config)
    composed_transform = transforms.Compose(transform_list)
    transformed_tensor_list = composed_transform(adj)

    print(transformed_tensor_list.shape)
    print(data_config.shape)


