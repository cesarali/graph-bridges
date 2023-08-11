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


def from_networkx_to_spins(graph_,upper_diagonal_indices,full_adjacency=False):
    adjacency_ = nx.to_numpy_array(graph_)
    if full_adjacency:
        spins = (-1.) ** (adjacency_.flatten() + 1)
    else:
        just_upper_edges = adjacency_[upper_diagonal_indices]
        spins = (-1.) ** (just_upper_edges.flatten() + 1)
    return spins

# Create a custom transformation class
class ToUpperDiagonalIndicesTransform:

    def __call__(self, tensor):
        if  len(tensor.shape) == 3:
            batch_size = tensor.shape[0]
            # Get the upper diagonal entries without zero-padding with the batch as the first dimension
            upper_diagonal_entries = tensor.masked_select(torch.triu(torch.ones_like(tensor), diagonal=1).bool())
            upper_diagonal_entries = upper_diagonal_entries.reshape(batch_size, -1)
            return upper_diagonal_entries
        else:
            raise Exception("Wrong Tensor Shape in Transform")

class FromUpperDiagonalTransform:

    def __call__(self, upper_diagonal_tensor):
        assert len(upper_diagonal_tensor.shape) == 2
        number_of_upper_entries = upper_diagonal_tensor.shape[1]
        batch_size = upper_diagonal_tensor.shape[0]

        matrix_size = int(.5 * (1 + np.sqrt(1 + 8 * number_of_upper_entries)))

        # Create a zero-filled tensor to hold the full matrices
        full_matrices = torch.zeros(batch_size, matrix_size, matrix_size, device=upper_diagonal_tensor.device)

        # Get the indices for the upper diagonal part of the matrices
        upper_tri_indices = torch.triu_indices(matrix_size, matrix_size, offset=1, device=upper_diagonal_tensor.device)

        # Fill the upper diagonal part of the matrices
        full_matrices[:, upper_tri_indices[0], upper_tri_indices[1]] = upper_diagonal_tensor

        # Transpose and fill the lower diagonal part to make the matrices symmetric
        full_matrices = full_matrices + full_matrices.transpose(1, 2)

        return full_matrices

class UnsqueezeTensorTransform:

    def __init__(self,axis=0):
        self.axis = axis
    def __call__(self, tensor:torch.Tensor):
        return tensor.unsqueeze(self.axis)

class BinaryTensorToSpinsTransform:

    def __call__(self,binary_tensor):
        spins = (-1.) ** (binary_tensor + 1)
        return spins


SqueezeTransform = transforms.Lambda(lambda x: x.squeeze())
FlattenTransform = transforms.Lambda(lambda x: x.reshape(x.shape[0], -1))
UnFlattenTransform = transforms.Lambda(lambda x: x.reshape(x.shape[0],
                                                           int(np.sqrt(x.shape[1])),
                                                           int(np.sqrt(x.shape[1]))))
SpinsToBinaryTensor = transforms.Lambda(lambda spins: ((1. + spins) * .5).float())
BinaryTensorToSpinsTransform = transforms.Lambda(lambda binary_tensor: (-1.) ** (binary_tensor + 1))

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

    return transform_list,inverse_transform_list

class BridgeGraphDataLoaders:
    """
    """
    graph_data_config : GraphDataConfig

    def __init__(self,config,device,type="data"):
        """

        :param config:
        :param device:
        """
        #config.max_node_num
        if type=="data":
            self.graph_data_config = config.data
        else:
            self.graph_data_config = config.target

        self.doucet = self.graph_data_config.doucet
        self.number_of_spins = self.graph_data_config.number_of_spins
        self.device = device

        transform_list,inverse_transform_list = get_transforms(self.graph_data_config)
        self.composed_transform = transforms.Compose(transform_list)
        self.transform_to_graph = transforms.Compose(inverse_transform_list)

        if self.graph_data_config.data == "MNIST" :
            train_graph_list, test_graph_list = self.read_pepperized_image_lists(self.graph_data_config.data )
        elif self.graph_data_config.data == "CIFAR":
            train_graph_list, test_graph_list = self.read_pepperized_image_lists(self.graph_data_config.data )
        else:
            train_graph_list, test_graph_list = self.read_graph_lists() 

        self.training_data_size = len(train_graph_list)
        self.test_data_size = len(test_graph_list)
        self.total_data_size = self.training_data_size + self.test_data_size

        train_adjs_tensor,train_x_tensor = self.graph_to_tensor_and_features(train_graph_list,
                                                                             self.graph_data_config.init,
                                                                             self.graph_data_config.max_node_num,
                                                                             self.graph_data_config.max_feat_num)

        train_adjs_tensor = self.composed_transform(train_adjs_tensor)

        self.train_dataloader_ = self.create_dataloaders(train_adjs_tensor,train_x_tensor)

        test_adjs_tensor,test_x_tensor = self.graph_to_tensor_and_features(test_graph_list,
                                                                           self.graph_data_config.init,
                                                                           self.graph_data_config.max_node_num,
                                                                           self.graph_data_config.max_feat_num)

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
        train_graph_list, test_graph_list = graph_list[test_size:], graph_list[:test_size]

        return train_graph_list, test_graph_list

    def read_pepperized_image_lists(self, image_set="MNIST"):
        data_dir = self.graph_data_config.dir
        threshold = self.graph_data_config.pepper_threshold
        pepperize = transforms.Compose([ transforms.ToTensor(),
                                        transforms.Lambda(lambda x: (x > threshold).float())
                                        ])
        if image_set=="MNIST":
            dataset = MNIST(root=data_dir, train=True, download=True, transform=pepperize)
        elif image_set=="CIFAR":
            dataset = CIFAR10(root=data_dir, train=True, download=True, transform=pepperize)
        test_size = int(self.graph_data_config.test_split * len(dataset))
        train_size = len(dataset) - test_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        return train_dataset.dataset, test_dataset.dataset


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



