import os
import torch
import pickle
import shutil
import numpy as np
import networkx as nx
from pprint import pprint
from torch import sigmoid
import torchvision

from typing import Union,Tuple,List
from torchtyping import TensorType

from abc import ABC, abstractmethod
from torch.distributions import Bernoulli
from graph_bridges.data.datasets import DictDataSet
from graph_bridges.data.dataloaders_utils import register_dataloader
from torch.utils.data import TensorDataset,DataLoader,random_split
from graph_bridges.configs.graphs.lobster.config_base import BridgeConfig

class BaseDataLoader(ABC):

    name_="base_data_loader"

    def __init__(self,config:BridgeConfig):
        super(BaseDataLoader,self).__init__()
        self.training_proportion = config.data.training_proportion
        self.batch_size = config.data.batch_size

    def define_dataset_and_dataloaders(self,X):
        if isinstance(X,torch.Tensor):
            dataset = TensorDataset(X)
        elif isinstance(X,dict):
            dataset = DictDataSet(X)

        self.total_data_size = len(dataset)
        self.training_data_size = int(self.training_proportion * self.total_data_size)
        self.test_data_size = self.total_data_size - self.training_data_size

        training_dataset, test_dataset = random_split(dataset, [self.training_data_size, self.test_data_size])
        self._train_iter = DataLoader(training_dataset, batch_size=self.batch_size)
        self._test_iter = DataLoader(test_dataset, batch_size=self.batch_size)

    def train(self):
        return self._train_iter

    def test(self):
        return self._test_iter

class BridgeData:

    config : BridgeConfig

    def __init__(self,config:BridgeConfig,device,rank=None):
        self.config = config
        self.device = self.config.device

        C,H,W = self.config.data.shape
        self.D = C*H*W
        self.S = self.config.data.S
        sampler_config = self.config.sampler

        self.initial_dist = sampler_config.initial_dist
        if self.initial_dist == 'gaussian':
            self.initial_dist_std  = self.config.model.Q_sigma
        else:
            self.initial_dist_std = None

    @abstractmethod
    def sample(self, num_of_paths:int, device=None) -> TensorType["num_of_paths","D"]:
        return None

@register_dataloader
class DoucetTargetData(BridgeData):

    def __init__(self,config:BridgeConfig,device,rank=None):
        BridgeData.__init__(self,config,device,rank)

    def sample(self, num_of_paths:int, device=None) -> TensorType["num_of_paths","D"]:
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
        else:
            raise NotImplementedError('Unrecognized initial dist ' + self.initial_dist)

        return x


if __name__=="__main__":
    from graph_bridges.configs.graphs.lobster.config_base import BridgeConfig
    from graph_bridges.data.dataloaders_utils import create_dataloader

    config = BridgeConfig()
    device = torch.device(config.device)
    data = create_dataloader(config,device,target=True)
    x = data.sample(num_of_paths=20,device=device)


