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
    estimator = get_loss(cfg.model.name)(cfg, device, rank)
    estimator = estimator.to(device)
    return estimator