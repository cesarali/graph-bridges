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
from graph_bridges.data.dataloaders_config import GraphSpinsDataLoaderConfig
from pathlib import Path
import numpy as np


data_path = Path(data_path)
graph_data_path = data_path / "raw" / "graph"

@dataclass
class EgoConfig:
    data: str =  "ego_small"
    dir: Path = graph_data_path
    batch_size: int = 128
    test_split: float = 0.2
    max_node_num: int = 18
    max_feat_num: int = 17
    init: str = "deg"

@dataclass
class CommunityConfig:
    data: str = 'community_small'
    dir: Path = graph_data_path
    batch_size: int = 128
    test_split: float = 0.2
    max_node_num: int = 20
    max_feat_num: int = 10
    init: str = 'deg'

@dataclass
class GridConfig:
    data: str = 'grid'
    dir: Path = graph_data_path
    batch_size: int = 8
    test_split: float = 0.2
    max_node_num: int = 361
    max_feat_num: int = 5
    init: str = 'deg'


@dataclass
class EnzymesConfig:
    data: str = 'ENZYMES'
    dir: Path = graph_data_path
    batch_size: int = 64
    test_split: float = 0.2
    max_node_num: int = 125
    max_feat_num: int = 10
    init: str = 'deg'


@dataclass
class QM9Config:
    data: str = 'QM9'
    dir: Path = graph_data_path
    batch_size: int = 1024
    max_node_num: int = 9
    max_feat_num: int = 4
    init: str = 'atom'

@dataclass
class ZincConfig:
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
    root : str = "datasets_folder"
    train : bool = True
    download : bool = True
    S : int = 2
    batch_size :int = 28 # use 128 if you have enough memory or use distributed
    shuffle : bool = True

    shape : List[int] = field(default_factory=lambda : [1,1,45])
    C: int = field(init=False)
    H: int = field(init=False)
    W: int = field(init=False)

    D :int = field(init=False)

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
