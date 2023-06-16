import os
import torch
import pickle
import shutil
import numpy as np
import networkx as nx
from pprint import pprint
from torch import sigmoid
import torchvision

from abc import ABC, abstractmethod
from torch.distributions import Bernoulli
from graph_bridges.data.datasets import DictDataSet
from torch.utils.data import TensorDataset,DataLoader,random_split


class BaseDataLoader(ABC):
    name_="base_data_loader"
    def __init__(self,**kwargs):
        super(BaseDataLoader,self).__init__()
        self.parameters_ = kwargs
        self.parameters_.update({"name":self.name_})

    def define_dataset_and_dataloaders(self,X,training_proportion=0.8,batch_size=32):
        self.batch_size = batch_size
        if isinstance(X,torch.Tensor):
            dataset = TensorDataset(X)
        elif isinstance(X,dict):
            dataset = DictDataSet(X)

        self.total_data_size = len(dataset)
        self.training_data_size = int(training_proportion * self.total_data_size)
        self.test_data_size = self.total_data_size - self.training_data_size

        training_dataset, test_dataset = random_split(dataset, [self.training_data_size, self.test_data_size])
        self._train_iter = DataLoader(training_dataset, batch_size=batch_size)
        self._test_iter = DataLoader(test_dataset, batch_size=batch_size)

    def train(self):
        return self._train_iter

    def test(self):
        return self._test_iter
