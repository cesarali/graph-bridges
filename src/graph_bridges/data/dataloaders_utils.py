from graph_bridges.configs.graphs.lobster.config_base import BridgeConfig
from graph_bridges.data.dataloaders_config import GraphSpinsDataLoaderConfig
from typing import Union

_DATALOADERS = {}

def register_dataloader(cls):
    name = cls.__name__
    if name in _DATALOADERS:
        raise ValueError(f'{name} is already registered!')
    _DATALOADERS[name] = cls
    return cls

def get_dataloader(name):
    return _DATALOADERS[name]


def create_dataloader(cfg:Union[GraphSpinsDataLoaderConfig,BridgeConfig],device,rank=None,target=False):
    if target:
        if isinstance(cfg.target,GraphSpinsDataLoaderConfig):
            dataloader = get_dataloader(cfg.target.name)(cfg.target, device, rank)
        else:
            dataloader = get_dataloader(cfg.target.name)(cfg, device, rank)
    else:
        if isinstance(cfg.data,GraphSpinsDataLoaderConfig):
            dataloader = get_dataloader(cfg.data.name)(cfg.data, device, rank)
        else:
            dataloader = get_dataloader(cfg.data.name)(cfg, device, rank)

    return dataloader