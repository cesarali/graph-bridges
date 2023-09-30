import os
import torch
from typing import List
from dataclasses import dataclass

from graph_bridges.data.spin_glass_dataloaders import ParametrizedSpinGlassHamiltonianLoader
from graph_bridges.data.graph_dataloaders import BridgeGraphDataLoaders,DoucetTargetData
from graph_bridges.data.image_dataloaders import DiscreteCIFAR10Dataloader
from graph_bridges.data.image_dataloaders import NISTLoader

def load_dataloader(config,type:str="data",device:torch.device=torch.device("cpu"),rank=None):
    if type == "data":
        if config.data.data in ['grid','community','community_small',"ego_small",'ENZYMES','QM9','ZINC250k']:
            dataloader = BridgeGraphDataLoaders(config.data,device)
        elif config.data.data in ['mnist','fashion','emnist']:
            dataloader = NISTLoader(config, device)
        elif config.data.data in ["Cifar10"]:
            dataloader = DiscreteCIFAR10Dataloader(config, device)
        elif config.data.data == "DoucetTargetData":
            dataloader = DoucetTargetData(config,device)
        elif config.data.name == "ParametrizedSpinGlassHamiltonian":
            dataloader = ParametrizedSpinGlassHamiltonianLoader(config.data, device)
        else:
            raise Exception("{0} not found in dataloaders".format(config.data.data))
    elif type == "target":
        if config.target.data in ['grid','community','community_small',"ego_small",'ENZYMES','QM9','ZINC250k']:
            dataloader = BridgeGraphDataLoaders(config.target,device)
        elif config.target.data in ['mnist','fashion','emnist']:
            dataloader = NISTLoader(config, device)
        elif config.target.data == "DoucetTargetData":
            dataloader = DoucetTargetData(config,device)
        elif config.target.name == "ParametrizedSpinGlassHamiltonian":
            dataloader = ParametrizedSpinGlassHamiltonianLoader(config.target, device)
        else:
            raise Exception("{0} not found in dataloaders".format(config.data.data))
    return dataloader


@dataclass
class DataBasics:

    D: int = None
    temporal_net_expected_shape: List[int] = None
    training_size: int = None
    test_size: int = None
    total_data_size: int = None
    def __post_init__(self):
        self.total_data_size = self.training_size + self.test_size

def check_sizes(sb_config)->DataBasics:
    sb_config.align_configurations()
    device = torch.device("cpu")
    data_dataloader = load_dataloader(sb_config, type="data", device=device)
    D = sb_config.data.D
    temporal_net_expected_shape = sb_config.data.temporal_net_expected_shape
    training_size = sb_config.data.training_size
    test_size = sb_config.data.test_size
    return DataBasics(D=D,temporal_net_expected_shape=temporal_net_expected_shape,training_size=training_size,test_size=test_size)
