import torch

from .graph_dataloaders import BridgeGraphDataLoaders
from .dataloaders import DoucetTargetData, GraphSpinsDataLoader

def load_dataloader(config,type:str="data",device:torch.device=torch.device("cpu"),rank=None):
    if type == "data":
        if config.data.data == "GraphSpinsDataLoader":
            dataloader = GraphSpinsDataLoader(config.data,device,rank)
        elif config.data.data in ['grid','community_small',"ego_small",'ENZYMES','QM9','ZINC250k']:
            dataloader = BridgeGraphDataLoaders(config,device)
        elif config.data.data == "DoucetTargetData":
            dataloader = DoucetTargetData(config,device)
        else:
            raise Exception("{0} not found in dataloaders".format(config.data.data))
    elif type == "target":
        if config.target.data == "GraphSpinsDataLoader":
            dataloader = GraphSpinsDataLoader(config.data,device,rank)
        elif config.target.data in ['grid','community_small',"ego_small",'ENZYMES','QM9','ZINC250k']:
            dataloader = BridgeGraphDataLoaders(config,device)
        elif config.target.data == "DoucetTargetData":
            dataloader = DoucetTargetData(config,device)
        else:
            raise Exception("{0} not found in dataloaders".format(config.data.data))
    return dataloader