import os
import torch
import pickle
import numpy as np
from typing import List,Dict,Tuple,Union
from torch.utils.data import TensorDataset, DataLoader
from graph_bridges.utils.graph_utils import init_features, graphs_to_tensor
from torchtyping import TensorType
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10

from graph_bridges.data.images_dataloaders_config import PepperMNISTConfig


def get_transforms(config:GraphDataConfig):
    """
    :param config:

    :return: transform_list,inverse_transform_list
    """
    SqueezeTransform = transforms.Lambda(lambda x: x.squeeze())
    FlattenTransform = transforms.Lambda(lambda x: x.reshape(x.shape[0], -1))
    UnFlattenTransform = transforms.Lambda(lambda x: x.reshape(x.shape[0],
                                                               int(np.sqrt(x.shape[1])),
                                                               int(np.sqrt(x.shape[1]))))

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
        transform_list.append(BinaryTensorToSpinsTransform())

    return transform_list,inverse_transform_list

class PepperMNISTDataLoaders:

    image_data_config : PepperMNISTDataConfig

    def __init__(self,config, device, type="data"):
        """

        :param config:
        :param device:
        """
        if type=="data": self.image_data_config = config.data
        else: self.image_data_config = config.target

        self.device = device

        transform_list, inverse_transform_list = get_transforms(self.image_data_config)

        self.composed_transform = transforms.Compose(transform_list)
        self.transform_to_graph = transforms.Compose(inverse_transform_list)

        train_graph_list, test_graph_list = self.read_pepper_mnist_lists()





        self.training_data_size = len(train_graph_list)
        self.test_data_size = len(test_graph_list)
        self.total_data_size = self.training_data_size + self.test_data_size

        train_adjs_tensor,train_x_tensor = self.graph_to_tensor_and_features(train_graph_list,
                                                                             self.image_data_config.init,
                                                                             self.image_data_config.max_node_num,
                                                                             self.image_data_config.max_feat_num)

        train_adjs_tensor = self.composed_transform(train_adjs_tensor)

        self.train_dataloader_ = self.create_dataloaders(train_adjs_tensor,train_x_tensor)

        test_adjs_tensor,test_x_tensor = self.graph_to_tensor_and_features(test_graph_list,
                                                                           self.image_data_config.init,
                                                                           self.image_data_config.max_node_num,
                                                                           self.image_data_config.max_feat_num)

        test_adjs_tensor = self.composed_transform(test_adjs_tensor)

        self.test_dataloader_ = self.create_dataloaders(test_adjs_tensor,test_x_tensor)

    def train(self):
        return self.train_dataloader_

    def test(self):
        return self.test_dataloader_

    def create_dataloaders(self,x_tensor, adjs_tensor):
        train_ds = TensorDataset(x_tensor, adjs_tensor)
        train_dl = DataLoader(train_ds,
                              batch_size=self.image_data_config.batch_size,
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

    def read_pepperized_mnist_lists(self)->Tuple[List[nx.Graph]]:
        data_dir = self.image_data_config.dir
        file_name = self.image_data_config.data
        file_path = os.path.join(data_dir, file_name)
        threshold = self.image_data_config.pepper_threshold
        pepperize = transforms.Compose([ transforms.ToTensor(),
                                        transforms.Lambda(lambda x: (x > threshold).float())
                                        ])
        mnist_list = MNIST(root='data', train=True, download=True, transform=pepperize)
        test_size = int(self.image_data_config.test_split * len(mnist_list))
        train_graph_list, test_graph_list = mnist_list[test_size:], mnist_list[:test_size]

        return train_graph_list, test_graph_list