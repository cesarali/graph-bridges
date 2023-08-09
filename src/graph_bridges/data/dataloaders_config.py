import os
from dataclasses import dataclass
import numpy as np
from graph_bridges import data_path
from typing import List,Union,Tuple,Optional,Dict
from dataclasses import field
from collections import defaultdict

def create_default_list() -> List[Optional[int]]:
    return []


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

if __name__=="__main__":
    from pprint import pprint
    config = GraphSpinsDataLoaderConfig()
    pprint(config)