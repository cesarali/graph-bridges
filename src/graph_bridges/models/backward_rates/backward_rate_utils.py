import torch
from typing import Union

from graph_bridges.models.backward_rates.ctdd_backward_rate import (
    BackRateMLP,
    GaussianTargetRateImageX0PredEMA,
    BackwardRateTemporalHollowTransformer
)
from graph_bridges.models.backward_rates.sb_backward_rate import SchrodingerBridgeBackwardRate

def load_backward_rates(config,device:torch.device):
    if config.model.name == "BackRateMLP":
        backward_rate = BackRateMLP(config,device)
    elif config.model.name == "GaussianTargetRateImageX0PredEMA":
        backward_rate = GaussianTargetRateImageX0PredEMA(config,device)
    elif config.model.name == "BackwardRateTemporalHollowTransformer":
        backward_rate = BackwardRateTemporalHollowTransformer(config,device)
    elif config.model.name == "SchrodingerBridgeBackwardRate":
        backward_rate = SchrodingerBridgeBackwardRate(config,device)
    else:
        raise Exception("{0} backward rate not implemented".format(config.model.name))

    return backward_rate
