import os
from pathlib import Path
from dataclasses import dataclass,field
from typing import List
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
    random_flips = True
    preprocess_datapath:str = "graphs"
    doucet:bool = True
    as_spins:bool = False

    total_data_size:int = 60000
    training_size:int = 50000
    test_size:int = 10000

    def __post_init__(self):

        self.shape = [3,32,32]
        self.temporal_net_expected_shape = self.shape
        self.D = self.C * self.H * self.W
        self.S = 256
        self.data_min_max = [0,255]

@dataclass
class NISTLoaderConfig:
    name:str = "NISTLoader"
    data:str = "mnist" # emnist, fashion, mnist
    dataloader_data_dir:str = None
    dir:str = None

    input_dim: int = 784
    batch_size: int = 32
    delete_data:bool = False
    pepper_threshold: float = 0.5

    total_data_size:int = 70000
    training_size:int = 60000
    test_size:int = 10000

    D = 784
    C = 1
    S = 2
    H = 28
    W = 28

    number_of_spins: int = 784
    number_of_states: int = 2

    as_image: bool = True
    as_spins: bool = False
    doucet: bool = True

    data_min_max:List[float] = field(default_factory=lambda:[0.,1.])

    def __post_init__(self):
        from graph_bridges import data_path
        self.dataloader_data_dir = os.path.join(data_path,"raw")
        self.dataloader_data_dir_file = os.path.join(self.dataloader_data_dir,self.data+".tr")
        self.dir = self.dataloader_data_dir_file
        self.preprocess_datapath = os.path.join(data_path,"raw",self.data)

        if self.as_spins:
            self.doucet = False

        if self.doucet:
            self.type = "doucet"

        if self.as_image:
            self.shape = [1, 28, 28]
            self.temporal_net_expected_shape = self.shape
            self.D = self.C * self.H * self.W
            self.data_min_max = [0, 1]
            self.S = 2
        else:
            #self.C = self.H, self.W
            self.shape = [None,None,None]
            self.temporal_net_expected_shape = [self.D]

        if self.as_spins:
            self.data_min_max = [-1.,1.]


