from pathlib import Path
from dataclasses import dataclass
from graph_bridges import data_path

data_path = Path(data_path)
image_data_path = data_path / "raw"

@dataclass
class DiscreteCIFAR10Config:
    data: str = "Cifar10"
    dir: Path=image_data_path
    batch_size: int= 16

    C: int = 3
    H: int = 32
    W: int = 32
    S: int = 256
    D: int = None

    shape: list = None
    random_flips = False
    preprocess_datapath:str = "graphs"

    def __post_init__(self):

        self.shape = [3,32,32]
        self.shape_ = self.shape
        self.D = self.C * self.H * self.W

        self.S = 2
        self.data_min_max = [0,256]
