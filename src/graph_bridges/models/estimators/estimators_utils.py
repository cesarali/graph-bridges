_ESTIMATOR = {}

def register_estimator(cls):
    name = cls.__name__
    if name in _ESTIMATOR:
        raise ValueError(f'{name} is already registered!')
    _ESTIMATOR[name] = cls
    return cls

def get_estimator(name):
    return _ESTIMATOR[name]

def create_model(cfg, device, rank=None):
    estimator = get_estimator(cfg.model.name)(cfg, device, rank)
    estimator = estimator.to(device)
    return estimator