_REFERENCE_PROCESS = {}

def register_reference(cls):
    name = cls.__name__
    if name in _REFERENCE_PROCESS:
        raise ValueError(f'{name} is already registered!')
    _REFERENCE_PROCESS[name] = cls
    return cls

def get_reference(name):
    return _REFERENCE_PROCESS[name]

def create_reference(cfg, device, rank=None):
    reference = get_reference(cfg.reference.name)(cfg, device, rank)
    reference = reference.to(device)
    return reference