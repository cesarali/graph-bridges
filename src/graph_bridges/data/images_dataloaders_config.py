from graph_bridges import data_path
from dataclasses import dataclass
from pathlib import Path
from dataclasses import dataclass
from pathlib import Path
from graph_bridges.data.graph_dataloaders_config import GraphDataConfig

data_path = Path(data_path)
image_data_path = data_path / "raw" 

@dataclass
class PepperMNISTDataConfig(GraphDataConfig):
    data: str =  "PEPPER-MNIST"
    dir: Path = image_data_path
    pepper_threshold: float = 0.5
    batch_size: int = 128
    test_split: float = 0.2
    max_node_num: int = 28
    max_feat_num: int = 28
    total_data_size: int = 60000
    init: str = "deg"

@dataclass
class PepperCIFARDataConfig(GraphDataConfig):    
    data: str =  "PEPPER-CIFAR"
    dir: Path = image_data_path
    pepper_threshold: float = 0.5
    batch_size: int = 128
    test_split: float = 0.2
    # max_node_num: int = 28
    # max_feat_num: int = 28
    # total_data_size: int = 60000
    # init: str = "deg"