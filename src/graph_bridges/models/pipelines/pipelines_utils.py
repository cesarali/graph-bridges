_PIPELINES = {}

def register_pipeline(cls):
    name = cls.__name__
    if name in _PIPELINES:
        raise ValueError(f'{name} is already registered!')
    _PIPELINES[name] = cls
    return cls

def get_pipelines(cfg):
    return _PIPELINES[cfg.scheduler.name](cfg)

def create_pipelines(cfg, device, rank=None):
    model = get_pipelines(cfg.pipeline.name)(cfg, device, rank)
    model = model.to(device)
    return model