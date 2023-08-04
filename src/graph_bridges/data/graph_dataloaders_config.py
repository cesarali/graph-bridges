from graph_bridges import data_path
from dataclasses import dataclass
from pathlib import Path
from dataclasses import dataclass,asdict,field
from typing import List, Union, Optional, Tuple, Dict
import shutil
import time
import os
import subprocess
import json
import numpy as np
from pathlib import Path

data_path = Path(data_path)
graph_data_path = data_path / "raw" / "graph"

@dataclass
class GraphDataConfig:
    data: str =None
    dir: Path=None
    batch_size: int=None
    test_split: float=None
    max_node_num: int=None
    max_feat_num: int=None
    init: str=None
    full_adjacency: bool = True
    flatten_adjacency: bool = True
    as_spins: bool= False
    as_image: bool= True
    C: int = None
    H: int = None
    W: int = None
    D: int = None
    S: int = None
    number_of_spins: int = None
    number_of_states: int = None
    total_data_size:int = 200
    training_size:int = None
    test_size:int = None


    shape : List[int] = None
    preprocess_datapath:str = "graphs"
    doucet:bool = False
    def __post_init__(self):
        self.number_of_upper_entries = int(self.max_node_num*(self.max_node_num-1)*.5)

        if self.flatten_adjacency:
            if self.full_adjacency:
                if self.as_image:
                    self.shape = [1, 1, self.max_node_num*self.max_node_num]
                    self.shape_ = self.shape
                    self.C, self.H, self.W = self.shape[0], self.shape[1], self.shape[2]
                    self.D = self.C * self.H * self.W
                else:
                    self.shape = [1,1,self.max_node_num * self.max_node_num]
                    self.shape_ = [self.max_node_num * self.max_node_num]
                    self.C, self.H, self.W = self.shape[0], self.shape[1], self.shape[2]
                    self.D = self.max_node_num * self.max_node_num
            else:
                if self.as_image:
                    self.shape = [1, 1,self.number_of_upper_entries]
                    self.shape_ = self.shape
                    self.C, self.H, self.W = self.shape[0], self.shape[1], self.shape[2]
                    self.D = self.C * self.H * self.W
                else:
                    self.shape = [1,1,self.number_of_upper_entries]
                    self.shape_ = [self.number_of_upper_entries]

                    self.C, self.H, self.W = self.shape[0], self.shape[1], self.shape[2]
                    self.D = self.number_of_upper_entries
        else:
            if self.full_adjacency:
                if self.as_image:
                    self.shape = [1,self.max_node_num,self.max_node_num]
                    self.shape_ = self.shape
                    self.C, self.H, self.W = self.shape[0], self.shape[1], self.shape[2]
                    self.D = self.C * self.H * self.W
                else:
                    self.shape_ = [self.max_node_num, self.max_node_num]
                    self.shape = [1,1,self.max_node_num, self.max_node_num]
                    self.H, self.W =  self.shape[0], self.shape[1]
                    self.C = None
                    self.D = self.max_node_num * self.max_node_num
            else: # no flatten no full adjacency
                raise Exception("No Flatten and No Full Adjacency incompatible for data")

        self.S = 2
        self.number_of_nodes = self.max_node_num
        self.number_of_spins = self.D
        self.number_of_states = self.S
        self.data_min_max = [0,1]
        if self.as_spins:
            self.doucet = False
        self.training_proportion = 1. - self.test_split
        self.training_size = int(self.training_proportion*self.total_data_size)
        self.test_size = int(self.test_split*self.total_data_size)

@dataclass
class EgoConfig(GraphDataConfig):
    data: str =  "ego_small"
    dir: Path = graph_data_path
    batch_size: int = 128
    test_split: float = 0.2
    max_node_num: int = 18
    max_feat_num: int = 17
    total_data_size:int = 200
    init: str = "deg"

@dataclass
class CommunityConfig(GraphDataConfig):
    data: str = 'community_small'
    dir: Path = graph_data_path
    batch_size: int = 128
    test_split: float = 0.2
    max_node_num: int = 20
    max_feat_num: int = 10
    init: str = 'deg'

@dataclass
class GridConfig(GraphDataConfig):
    data: str = 'grid'
    dir: Path = graph_data_path
    batch_size: int = 8
    test_split: float = 0.2
    max_node_num: int = 361
    max_feat_num: int = 5
    init: str = 'deg'

@dataclass
class EnzymesConfig(GraphDataConfig):
    data: str = 'ENZYMES'
    dir: Path = graph_data_path
    batch_size: int = 64
    test_split: float = 0.2
    max_node_num: int = 125
    max_feat_num: int = 10
    init: str = 'deg'

@dataclass
class QM9Config(GraphDataConfig):
    data: str = 'QM9'
    dir: Path = graph_data_path
    batch_size: int = 1024
    max_node_num: int = 9
    max_feat_num: int = 4
    init: str = 'atom'

@dataclass
class ZincConfig(GraphDataConfig):
    data: str = 'ZINC250k'
    dir: Path = graph_data_path
    batch_size: int = 1024
    max_node_num: int = 38
    max_feat_num: int = 9
    init: str = 'atom'

@dataclass
class TargetConfig:
    # doucet variables
    name : str = 'DoucetTargetData'
    data : str = 'DoucetTargetData'

    root : str = "datasets_folder"
    train : bool = True
    download : bool = True
    S : int = 2
    batch_size :int = 28 # use 128 if you have enough memory or use distributed
    shuffle : bool = True

    shape : List[int] = field(default_factory=lambda : [1,1,45])
    C: int = None
    H: int = None
    W: int = None
    D :int = None

    random_flips : int = True

    # discrete diffusion variables
    type : str = "doucet" #one of [doucet, spins]
    full_adjacency : bool = False
    preprocess_datapath :str = "lobster_graphs_upper"
    raw_datapath :str = "lobster_graphs_upper"

    #length = 500
    max_node : int = 10
    min_node : int = 10

    def __post_init__(self):
        self.C, self.H, self.W = self.shape[0], self.shape[1], self.shape[2]
        self.D = self.C * self.H * self.W

@dataclass
class GraphSpinsDataLoaderConfig:
    name:str = "GraphSpinsDataLoader"
    data:str = "GraphSpinsDataLoader"
    graph_type:str = 'lobster'
    remove:bool = False
    training_proportion:float = 0.8
    doucet:bool = True
    data_path:str = ""

    S:int = 2
    shape : List[int] = field(default_factory=lambda:[1,1,45])
    possible_params_dict: Dict[str, List[Optional[float]]] = field(default_factory=lambda : {"k":None})

    random_flips :bool = True
    data_min_max : List[int] = field(default_factory=lambda:[0, 1]) # CHECK THIS for CIFAR 255

    type:str = "doucet" #one of [doucet, spins]
    full_adjacency:bool = False

    number_of_paths : int = 500
    length :int = 500

    corrupt_func = Optional
    max_node : int = 10
    min_node : int = 10
    full_adjacency : bool = False

    number_of_nodes : int = 10
    number_of_spins : int = 100
    batch_size : int = 28

    def __post_init__(self):
        self.C, self.H, self.W = self.shape[0], self.shape[1], self.shape[2]
        self.D = self.C * self.H * self.W
        self.data_path = os.path.join(data_path, "raw", "graph", self.graph_type)
        self.number_of_spins = self.number_of_nodes * self.number_of_nodes
        self.length = self.number_of_paths

        self.possible_params_dict = {'n': np.arange(5, 16).tolist(),
                                     'p1': [0.7],
                                     'p2': [0.5]}


all_dataloaders_configs = {"ego_small":EgoConfig,
                           "community_small":CommunityConfig,
                           "grid":GridConfig,
                           "ENZYMES":EnzymesConfig,
                           "QM9":QM9Config,
                           "ZINC250k":ZincConfig,
                           "GraphSpinsDataLoader":GraphSpinsDataLoaderConfig,
                           "DoucetTargetData":TargetConfig}


if __name__=="__main__":

    #================================
    # upper half adjacency
    #================================

    graph_config = EgoConfig(full_adjacency=False,flatten_adjacency=True,as_image=True)
    print(graph_config.shape)

    graph_config = EgoConfig(full_adjacency=False,flatten_adjacency=True,as_image=False)
    print(graph_config.shape)

    # ================================
    # full matrix
    # ================================

    graph_config = EgoConfig(full_adjacency=True,flatten_adjacency=True, as_image=True)
    print(graph_config.shape)

    graph_config = EgoConfig(full_adjacency=True,flatten_adjacency=True, as_image=False)
    print(graph_config.shape)

    graph_config = EgoConfig(full_adjacency=True,flatten_adjacency=False, as_image=True)
    print(graph_config.shape)

    graph_config = EgoConfig(full_adjacency=True,flatten_adjacency=False, as_image=False)
    print(graph_config.shape)