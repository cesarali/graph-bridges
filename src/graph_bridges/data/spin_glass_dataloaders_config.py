import os
from typing import List
from dataclasses import dataclass, asdict,field
from graph_bridges import data_path

@dataclass
class SpinGlassVariablesConfig:
    #ISING VARIABLES
    number_of_spins : int = 4
    beta : int = 1.
    obtain_partition_function : bool = True
    couplings_deterministic : float = 1.
    couplings_sigma : float = 1.
    couplings : List[float] = None
    fields : List[float] = None
    number_of_paths : int = 800
    number_of_mcmc_steps : int = 1000
    number_of_mcmc_burning_steps: int = 500


@dataclass
class ParametrizedSpinGlassHamiltonianConfig(SpinGlassVariablesConfig):

    #NAMES
    name: str = "ParametrizedSpinGlassHamiltonian"
    data: str = "bernoulli_spins"#spin_glass, ising "bernoulli_spins_0.2","bernoulli_spins_0.8"

    bernoulli_spins:bool = False
    bernoulli_probability:float = 0.2

    delete_data:bool = False

    dataloader_data_dir:str = None
    dataloader_data_path:str = None

    dir: str = None
    batch_size: int = 32
    test_split: float = 0.2

    # CTDD or SB variables
    total_data_size:int = None
    training_size:int = None
    test_size:int = None

    as_spins: bool= False
    as_image: bool= False
    doucet:bool = False
    type:str=None

    C: int = None
    H: int = None
    W: int = None
    D: int = None
    S: int = 2

    data_min_max: List[float] = field(default_factory=lambda :[-1.,1.])

    def __post_init__(self):
        self.D  = self.number_of_spins
        self.number_of_states = 2

        if self.as_spins:
            self.doucet = False

        if self.doucet:
            self.type = "doucet"

        if self.as_image:
            self.C = 1
            self.H = 1
            self.W = self.D
            self.shape = [self.C,self.H,self.W]
            self.temporal_net_expected_shape = [self.C, self.H, self.W]
        else:
            #self.C = self.H, self.W
            self.shape = [None,None,None]
            self.temporal_net_expected_shape = [self.D]

        if not self.as_spins:
            self.data_min_max = [0.,1.]

        self.dataloader_data_dir = os.path.join(data_path,"raw","spin_glass")
        self.dataloader_data_path = os.path.join(self.dataloader_data_dir,f"{self.data}.pkl")
        self.preprocess_datapath = self.dataloader_data_dir
