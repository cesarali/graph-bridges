from graph_bridges.configs.graphs.lobster.config_base import BridgeConfig

_DATALOADERS = {}

def register_dataloader(cls):
    name = cls.__name__
    if name in _DATALOADERS:
        raise ValueError(f'{name} is already registered!')
    _DATALOADERS[name] = cls
    return cls

def get_dataloader(name):
    return _DATALOADERS[name]

def create_dataloader(cfg: BridgeConfig,device,rank=None,target=False):
    if target:
        dataloader = get_dataloader(cfg.target.name)(cfg, device, rank)
    else:
        dataloader = get_dataloader(cfg.data.name)(cfg, device, rank)
    return dataloader