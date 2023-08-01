import torch

from graph_bridges.models.backward_rates.backward_rate import BackRateMLP, GaussianTargetRateImageX0PredEMA
from typing import Union


def load_backward_rates(config,device:torch.device):
    if config.model.name == "BackRateMLP":
        backward_rate = BackRateMLP(config,device)
    elif config.model.name == "GaussianTargetRateImageX0PredEMA":
        backward_rate = GaussianTargetRateImageX0PredEMA(config,device)
    else:
        raise Exception("{0} backward rate not implemented".format(config.model.name))

    return backward_rate
