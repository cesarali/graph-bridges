import os
import torch
import pickle
import numpy as np
import networkx as nx
from typing import List,Dict,Tuple,Union
from torch.utils.data import TensorDataset, DataLoader
from graph_bridges.utils.graph_utils import init_features, graphs_to_tensor
from torchtyping import TensorType
from graph_bridges.data.graph_dataloaders_config import CommunityConfig, GraphDataConfig
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from abc import abstractmethod
from graph_bridges.configs.graphs.graph_config_ctdd import CTDDConfig
from graph_bridges.configs.graphs.graph_config_sb import SBConfig

from graph_bridges.data.transforms import (
    FlattenTransform,
    UnsqueezeTensorTransform,
    SqueezeTransform,
    UnFlattenTransform,
    FromUpperDiagonalTransform,
    ToUpperDiagonalIndicesTransform,
    BinaryTensorToSpinsTransform,
    SpinsToBinaryTensor
)

def from_networkx_to_spins(graph_,upper_diagonal_indices,full_adjacency=False):
    adjacency_ = nx.to_numpy_array(graph_)
    if full_adjacency:
        spins = (-1.) ** (adjacency_.flatten() + 1)
    else:
        just_upper_edges = adjacency_[upper_diagonal_indices]
        spins = (-1.) ** (just_upper_edges.flatten() + 1)
    return spins

def get_transforms(config:GraphDataConfig):
    """
    :param config:

    :return: transform_list,inverse_transform_list
    """

    if config.flatten_adjacency:
        if config.full_adjacency:
            if config.as_image:
                transform_list = [FlattenTransform,UnsqueezeTensorTransform(1),UnsqueezeTensorTransform(1)]
                inverse_transform_list = [SqueezeTransform,UnFlattenTransform]
            else:
                transform_list = [FlattenTransform]
                inverse_transform_list = [UnFlattenTransform]
        else:
            if config.as_image:
                transform_list = [ToUpperDiagonalIndicesTransform(), UnsqueezeTensorTransform(1), UnsqueezeTensorTransform(1)]
                inverse_transform_list = [SqueezeTransform,FromUpperDiagonalTransform()]
            else:
                transform_list = [ToUpperDiagonalIndicesTransform()]
                inverse_transform_list = [FromUpperDiagonalTransform()]
    else:
        if config.full_adjacency:
            if config.as_image:
                transform_list = [UnsqueezeTensorTransform(1)]
                inverse_transform_list = [SqueezeTransform]
            else:
                transform_list = []
                inverse_transform_list = []
        else:  # no flatten no full adjacency
            raise Exception("No Flatten and No Full Adjacency incompatible for data")

    if config.as_spins:
        transform_list.append(BinaryTensorToSpinsTransform)
        inverse_transform_list.append(SpinsToBinaryTensor())

    return transform_list,inverse_transform_list

class BridgeGraphDataLoaders:
    """
    """
    graph_data_config : GraphDataConfig

    def __init__(self,graph_data_config,device):
        """

        :param config:
        :param device:
        """
        self.graph_data_config = graph_data_config

        self.doucet = self.graph_data_config.doucet
        self.number_of_spins = self.graph_data_config.number_of_spins
        self.device = device

        transform_list,inverse_transform_list = get_transforms(self.graph_data_config)
        self.composed_transform = transforms.Compose(transform_list)
        self.transform_to_graph = transforms.Compose(inverse_transform_list)

        train_graph_list, test_graph_list = self.read_graph_lists()

        self.training_data_size = len(train_graph_list)
        self.test_data_size = len(test_graph_list)
        self.total_data_size = self.training_data_size + self.test_data_size

        self.graph_data_config.training_size = self.training_data_size
        self.graph_data_config.test_size = self.test_data_size
        self.graph_data_config.total_data_size = self.total_data_size
        self.graph_data_config.training_proportion = float(self.training_data_size)/self.total_data_size

        train_adjs_tensor,train_x_tensor = self.graph_to_tensor_and_features(train_graph_list,
                                                                             self.graph_data_config.init,
                                                                             self.graph_data_config.max_node_num,
                                                                             self.graph_data_config.max_feat_num)
        test_adjs_tensor, test_x_tensor = self.graph_to_tensor_and_features(test_graph_list,
                                                                            self.graph_data_config.init,
                                                                            self.graph_data_config.max_node_num,
                                                                            self.graph_data_config.max_feat_num)


        train_adjs_tensor = self.composed_transform(train_adjs_tensor)
        self.train_dataloader_ = self.create_dataloaders(train_adjs_tensor,train_x_tensor)

        test_adjs_tensor = self.composed_transform(test_adjs_tensor)
        self.test_dataloader_ = self.create_dataloaders(test_adjs_tensor,test_x_tensor)

        self.fake_time_ = torch.rand(self.graph_data_config.batch_size)

    def train(self):
        return self.train_dataloader_

    def test(self):
        return self.test_dataloader_

    def sample(self,sample_size=10,type="train"):
        if type == "train":
            data_iterator = self.train()
        else:
            data_iterator = self.test()

        included = 0
        x_adj_list = []
        x_features_list = []
        for databatch in data_iterator:
            x_adj = databatch[0]
            x_features = databatch[1]
            x_adj_list.append(x_adj)
            x_features_list.append(x_features)

            current_batchsize = x_adj.shape[0]
            included += current_batchsize
            if included > sample_size:
                break

        if included < sample_size:
            raise Exception("Sample Size Smaller Than Expected")

        x_adj_list = torch.vstack(x_adj_list)
        x_features_list = torch.vstack(x_features_list)

        return [x_adj_list[:sample_size],x_features_list[:sample_size]]

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

        all_node_numbers = list(map(lambda x: x.number_of_nodes(), graph_list))

        max_number_of_nodes = max(all_node_numbers)
        min_number_of_nodes = min(all_node_numbers)

        train_graph_list, test_graph_list = graph_list[test_size:], graph_list[:test_size]

        return train_graph_list, test_graph_list


class DoucetTargetData():
    config : Union[SBConfig,CTDDConfig]

    doucet:bool = True

    def __init__(self,config:CTDDConfig,device,rank=None):
        self.config = config
        self.device = device
        self.doucet = self.config.data.doucet
        self.as_spins = self.config.data.as_spins
        self.config.target.training_size = self.config.data.training_size
        self.config.target.test_size = self.config.data.test_size
        self.config.target.total_data_size = self.config.data.total_data_size

        self.D = self.config.data.D
        self.number_of_spins = self.D
        self.shape = self.config.data.shape
        self.shape_ = self.config.data.temporal_net_expected_shape

        self.S = self.config.data.S
        sampler_config = self.config.sampler

        self.initial_dist = sampler_config.initial_dist
        if self.initial_dist == 'gaussian':
            self.initial_dist_std = self.config.reference.Q_sigma
        else:
            self.initial_dist_std = None

    def sample(self, num_of_paths:int, device=None) -> TensorType["num_of_paths","D"]:
        from graph_bridges.data.graph_dataloaders import BinaryTensorToSpinsTransform

        if device is None:
            device = self.device

        if self.initial_dist == 'uniform':
            x = torch.randint(low=0, high=self.S, size=(num_of_paths, self.D), device=device)
        elif self.initial_dist == 'gaussian':
            target = np.exp(
                - ((np.arange(1, self.S + 1) - self.S // 2) ** 2) / (2 * self.initial_dist_std ** 2)
            )
            target = target / np.sum(target)

            cat = torch.distributions.categorical.Categorical(
                torch.from_numpy(target)
            )
            x = cat.sample((num_of_paths * self.D,)).view(num_of_paths, self.D)
            x = x.to(device)
            if self.config.data.as_spins:
                x = BinaryTensorToSpinsTransform(x.float())
        else:
            raise NotImplementedError('Unrecognized initial dist ' + self.initial_dist)

        return [x,None]

    def train(self):
        #from graph_bridges.data.graph_dataloaders import BinaryTensorToSpinsTransform
        try:
            training_size = self.config.data.training_size
        except:
            training_size = int(self.config.data.total_data_size*self.config.data.training_proportion)

        batch_size = self.config.data.batch_size
        current_index = 0
        while current_index < training_size:
            remaining = min(training_size - current_index, batch_size)
            x = self.sample(remaining)
            # Your processing code here
            current_index += remaining
            yield x

    def test(self):
        try:
            test_size = self.config.data.test_size
        except:
            test_size = self.config.data.total_data_size - int(self.config.data.total_data_size*self.config.data.training_proportion)
        batch_size =  self.config.data.batch_size
        number_of_batches = int(test_size / batch_size) + 1
        for a in range(number_of_batches):
            x = self.sample(batch_size)
            yield x

# delete below?

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



