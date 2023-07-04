import torch
from graph_bridges.models.generative_models.ctdd import CTDD
from graph_bridges.configs.graphs.lobster.config_base import get_config_from_file

if __name__=="__main__":

    config = get_config_from_file("graph","lobster","1687884918")
    device = torch.device(config.device)
    ctdd = CTDD()
    ctdd.create_from_config(config,device)

    x = ctdd.pipeline(ctdd.model)
