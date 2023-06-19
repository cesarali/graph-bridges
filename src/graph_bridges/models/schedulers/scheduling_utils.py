_SCHEDULERS = {}

def register_scheduler(cls):
    name = cls.__name__
    if name in _SCHEDULERS:
        raise ValueError(f'{name} is already registered!')
    _SCHEDULERS[name] = cls
    return cls

def get_scheduler(name):
    return _SCHEDULERS[name]

def create_scheduler(cfg, device, rank=None):
    model = get_scheduler(cfg.scheduler.name)(cfg, device, rank)
    model = model.to(device)
    return model