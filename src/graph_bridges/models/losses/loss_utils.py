_LOSS = {}

def register_loss(cls):
    name = cls.__name__
    if name in _LOSS:
        raise ValueError(f'{name} is already registered!')
    _LOSS[name] = cls
    return cls

def get_loss(name):
    return _LOSS[name]

def create_loss(cfg, device, rank=None):
    loss = get_loss(cfg.loss.name)(cfg, device, rank)
    loss = loss.to(device)
    return loss