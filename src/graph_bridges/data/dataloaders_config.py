import os
from dataclasses import dataclass
import numpy as np
from graph_bridges import data_path

@dataclass
class GraphSpinsDataLoaderConfig:
    name = "GraphSpinsDataLoader"
    graph_type = 'lobster'
    remove = False
    training_proportion = 0.8
    doucet = True
    data_path = os.path.join(data_path,"raw","graph",graph_type)

    possible_params_dict =  {'n': np.arange(5, 16).tolist(),
                             'p1': [0.7],
                             'p2': [0.5]}

    number_of_paths = 100
    length = number_of_paths

    corrupt_func = None
    max_node = 10
    min_node = 10
    full_adjacency = False

    number_of_nodes = 10
    number_of_spins = number_of_nodes*number_of_nodes
    batch_size = 32
