import os
import torch
import pickle
import shutil
import numpy as np
import networkx as nx
from pprint import pprint
from torch import sigmoid
import torchvision

from typing import Union,Tuple,List,Optional
from torchtyping import TensorType

from abc import ABC, abstractmethod
from torch.distributions import Bernoulli
from graph_bridges.data.datasets import DictDataSet
from torch.utils.data import TensorDataset,DataLoader,random_split
from graph_bridges.configs.graphs.config_sb import SBConfig
from graph_bridges.utils.spin_utils import bool_to_spins, spins_to_bool,flip_and_copy_bool
from graph_bridges.data.graph_generators import gen_graph_list
from dataclasses import dataclass

from graph_bridges.configs.graphs.config_ctdd import CTDDConfig
from graph_bridges.configs.graphs.config_sb import SBConfig

from graph_bridges.data.dataloaders_config import GraphSpinsDataLoaderConfig


class BaseDataLoader(ABC):

    name_="base_data_loader"

    def __init__(self,cfg:Union[GraphSpinsDataLoaderConfig,SBConfig],device:torch.device,rank:int,X:torch.Tensor=None):
        if isinstance(cfg,BridgeConfig):
            assert  BridgeConfig.data.name == "GraphSpinsDataLoader"
            cfg = BridgeConfig.data
        if isinstance(cfg,GraphSpinsDataLoaderConfig):
            cfg = cfg

        super(BaseDataLoader,self).__init__()
        self.training_proportion = cfg.training_proportion
        self.batch_size = cfg.batch_size

    def define_dataset_and_dataloaders(self,X,training_proportion=None,batch_size=None):
        if training_proportion is not None:
            self.training_proportion=training_proportion
        if batch_size is not None:
            self.batch_size = batch_size

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

class SpinsDataLoader(BaseDataLoader):

    name_ = "spins"
    def __init__(self,cfg:Union[GraphSpinsDataLoaderConfig,SBConfig],device,rank,X=None):
        """
        If given a spins tensor sets as data
        If given a path string read data
        Otherwise it will sample and store the data in a predifined raw data folder

        :param X: torch.Tensor(number_of_paths,number_of_spins)
        :param training_proportion: float
        :param kwargs:
        """
        super(SpinsDataLoader,self).__init__(cfg,device,rank,X)
        if isinstance(cfg,BridgeConfig):
            assert  BridgeConfig.data.name == "GraphSpinsDataLoader"
            cfg = BridgeConfig.data
        if isinstance(cfg,GraphSpinsDataLoaderConfig):
            cfg = cfg

        self.number_of_spins = cfg.number_of_spins
        self.number_of_paths = cfg.number_of_paths

        training_proportion = cfg.training_proportion
        batch_size = cfg.batch_size
        self.doucet = cfg.doucet

        # data provided
        if X is not None:
            self.check_data(X)
            self.real_distribution = None
        else:
            data_path = cfg.data_path
            exists, data_path = self.checks_standard_file(data_path,cfg.remove)
            if exists:
                # data in file
                X, self.real_distribution_parameters = self.read_data(data_path)
                self.check_data(X)
            else:
                # data simulated
                print("Simulating {0} Data".format(self.name_))
                X, self.real_distribution_parameters = self.simulate_data(cfg)
                self.check_data(X)
                self.save_data({"X":X,
                                "real_distribution":self.real_distribution_parameters},
                               data_path)

        if self.doucet:
            X = self.convert_to_doucet(X)

        self.define_dataset_and_dataloaders(X,
                                            training_proportion=training_proportion,
                                            batch_size=batch_size)

        self.define_real_distribution()

    def check_data(self,X):
        if self.name_ == "marginal_ising":
            expected_dimension = 3
        else:
            expected_dimension = 2

        if isinstance(X,torch.Tensor):
            data_unique = set(X.reshape(-1).tolist())
            assert data_unique == set([1.,-1.]) #check only spins
            assert len(X.shape) == expected_dimension # check batch size and number of spins
            self.number_of_paths = X.shape[0]
            self.number_of_spins = X.shape[expected_dimension-1]
        else:
            raise Exception("Wrong Data Format")

    def convert_to_doucet(self,X):
        X[torch.where(X == -1.)] = 0.
        X = X.to(torch.int8)
        X = X.unsqueeze(1)
        X = X.unsqueeze(1)
        return X

    def checks_standard_file(self,data_path=None,remove=False):
        """
        checks if given  path or standard data path as defined by experiments string exists
        or requieres removal (hence does not exist)

        :param remove:
        :return: exists, data_path
        """
        from graph_bridges import data_path as parent_data_path

        # if data path is not provided, the standard path file is defined according to
        # the dataloaders name

        if data_path is None:
            data_folder = os.path.join(parent_data_path, "raw", self.name_)
            data_path = os.path.join(data_folder, "{0}.cp".format(self.name_))
        else:
            data_folder = os.path.join(parent_data_path, "raw", data_path)
            data_path = os.path.join(data_folder, "{0}.cp".format(self.name_))

        exists = os.path.exists(data_path)
        if exists and remove:
            shutil.rmtree(data_folder)
            exists = False

        if not exists:
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)

        return exists, data_path

    def save_data(self,X,data_path):
        """
        If data path not given, checks for standard save file and stored there

        :param X:
        :param data_path:
        :param remove:
        :return:
        """
        with open(data_path,"wb") as f:
            pickle.dump(X,f)

    def read_data(self,data_path):
        if os.path.exists(data_path):
            self.data_path = data_path
            with open(self.data_path, "rb") as f:
                data = pickle.load(f)
                X = data.get("X")
                real_distribution = data.get("real_distribution")
                self.check_data(X)
                return X, real_distribution
        else:
            raise Exception("Spin File Does Not Exist")

    @abstractmethod
    def simulate_data(self, **kwargs):
        return None

    def flip_and_copy_spins(self,X_spins):
        X_bool = spins_to_bool(X_spins)
        X_copy_bool, X_flipped_bool = flip_and_copy_bool(X_bool)
        X_copy_spin, X_flipped_spin = bool_to_spins(X_copy_bool), bool_to_spins(X_flipped_bool)
        return X_copy_spin, X_flipped_spin

    def define_real_distribution(self):
        return None

    def log_unnormalized(self,X_spins):
        return None

    def exact_flip_ratio(self, X_spins, dimension=1):
        """
        :param X_spins: torch.Tensor(batch_size,number_of_spins) or

        :return:
        """
        # EVALUATION
        batch_size = X_spins.shape[0]
        number_of_spins = X_spins.shape[1]

        if dimension is not None:
            assert dimension < number_of_spins

        X_copy_spin, X_flipped_spin = self.flip_and_copy_spins(X_spins)
        log_unnnormalized_X_copy = self.log_unnormalized(X_copy_spin)
        log_unnnormalized_X_flipped = self.log_unnormalized(X_copy_spin)
        log_ratio = log_unnnormalized_X_copy - log_unnnormalized_X_flipped
        log_ratio = log_ratio.reshape(batch_size,number_of_spins)
        probability_ratio = torch.exp(log_ratio)

        if dimension is not None:
            probability_ratio = probability_ratio[:,dimension]

        assert not torch.isnan(probability_ratio).any()
        assert not torch.isinf(probability_ratio).any()

        return probability_ratio

class GraphSpinsDataLoader(SpinsDataLoader):
    """
    databatch = next(dataloader.train().__iter__())
    X_spins = databatch[0]
    """
    name_ = "graph_spins"

    def __init__(self, cfg:Union[GraphSpinsDataLoaderConfig,SBConfig], device, rank):
        if isinstance(cfg,BridgeConfig):
            assert cfg.data.name == "GraphSpinsDataLoader"
            cfg = BridgeConfig.data
        if isinstance(cfg,GraphSpinsDataLoaderConfig):
            cfg = cfg
        self.graph_type = cfg.graph_type
        self.name_ = self.name_ + "_" + cfg.graph_type
        self.full_adjacency = cfg.full_adjacency
        self.number_of_nodes = cfg.number_of_nodes
        self.number_of_paths = cfg.number_of_paths

        if not self.full_adjacency:
            self.number_of_spins = np.triu_indices(self.number_of_nodes, k=1)[0].shape[0]
        else:
            self.number_of_spins = self.number_of_nodes ** 2
        cfg.number_of_spins = self.number_of_spins

        self.upper_diagonal_indices = np.triu_indices(self.number_of_nodes, k=1)
        super(GraphSpinsDataLoader, self).__init__(cfg,device,rank)

    def from_networkx_to_spins(self, graph_):
        adjacency_ = nx.to_numpy_array(graph_)
        if self.full_adjacency:
            spins = (-1.) ** (adjacency_.flatten() + 1)
        else:
            just_upper_edges = adjacency_[self.upper_diagonal_indices]
            spins = (-1.) ** (just_upper_edges.flatten() + 1)
        return spins

    def from_spins_to_networkx(self, X_spins):
        if not self.full_adjacency:
            graph_list = []
            X_spins[X_spins == -1.] = 0.
            batch_size = X_spins.shape[0]
            adjacencies = torch.zeros((batch_size,
                                       self.number_of_nodes,
                                       self.number_of_nodes))
            adjacencies[:,
            self.upper_diagonal_indices[0],
            self.upper_diagonal_indices[1]] = X_spins
            adjacencies = adjacencies + adjacencies.permute(0, 2, 1)
            adjacencies = adjacencies.numpy()
            for graph_index in range(batch_size):
                graph_list.append(nx.from_numpy_array(adjacencies[graph_index]))
            return graph_list
        else:
            graph_list = []
            batch_size = X_spins.shape[0]
            X_spins[X_spins == -1.] = 0.
            adjacencies = X_spins.numpy()
            for graph_index in range(batch_size):
                graph_list.append(nx.from_numpy_array(adjacencies[graph_index]))
            return graph_list

    def simulate_data(self, cfg:Union[SBConfig,GraphSpinsDataLoaderConfig]):
        if isinstance(cfg,BridgeConfig):
            cfg = cfg.data
        if isinstance(cfg,GraphSpinsDataLoaderConfig):
            cfg = cfg

        networkx_graphs = gen_graph_list(cfg.graph_type,
                                         cfg.possible_params_dict,
                                         cfg.corrupt_func,
                                         cfg.length,
                                         cfg.max_node,
                                         cfg.min_node)
        X = []
        for graph_ in networkx_graphs:
            print(graph_.number_of_nodes())
            x = self.from_networkx_to_spins(graph_)
            X.append(x[None, :])
        X = np.concatenate(X, axis=0)
        X = torch.Tensor(X)
        return X, (None,)

    # =======================================
    # EXACT VALUE OF ESTIMATOR
    # =======================================
    def define_real_distribution(self):
        """
        :return:
        """
        # self.bernoulli_real = Bernoulli(self.real_distribution_parameters[0])
        # raise Exception("Not Implemented")
        print("Real Distribution Not Implemented for {0}".format(self.name_))
        return None

class BridgeDataLoader:

    config : SBConfig
    doucet: bool = True

    def __init__(self,config:SBConfig,device,rank=None):
        self.config = config
        self.device = device

        C,H,W = self.config.data.shape

        self.D = C*H*W
        self.number_of_spins = self.D

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

class DoucetTargetData(BridgeDataLoader):
    doucet:bool = True
    def __init__(self,config:CTDDConfig,device,rank=None):
        BridgeDataLoader.__init__(self, config, device, rank)

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

    def train(self):
        self.config.data.training_proportion
        training_size = int(self.config.data.total_data_size*self.config.data.training_proportion)
        batch_size =  self.config.data.batch_size
        number_of_batches = int(training_size / batch_size)
        for a in range(number_of_batches):
            x = [self.sample(batch_size)]
            yield x

    def test(self):
        test_size = self.config.data.total_data_size - int(self.config.data.total_data_size*self.config.data.training_proportion)
        batch_size =  self.config.data.batch_size
        number_of_batches = int(test_size / batch_size) + 1
        for a in range(number_of_batches):
            x = [self.sample(batch_size)]
            yield x


all_dataloaders = {"GraphSpinsDataLoader":GraphSpinsDataLoader,
                   "DoucetTargetData":DoucetTargetData}

if __name__=="__main__":
    from graph_bridges.configs.graphs.lobster.config_base import BridgeConfig
    from graph_bridges.data.dataloaders_utils import create_dataloader
    from graph_bridges.data.dataloaders_config import GraphSpinsDataLoaderConfig

    config = BridgeConfig()
    config.data = GraphSpinsDataLoaderConfig()

    device = torch.device(config.device)

    dataloader = create_dataloader(config,device,target=False)

    #x = data.sample(num_of_paths=20,device=device)


