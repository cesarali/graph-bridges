import os
import torch

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
        elif config.data.data in ["spin_glass","ising","small_spins"]:
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
        elif config.target.data in ["spin_glass","ising","small_spins"]:
            dataloader = ParametrizedSpinGlassHamiltonianLoader(config.target, device)
        else:
            raise Exception("{0} not found in dataloaders".format(config.data.data))
    return dataloader