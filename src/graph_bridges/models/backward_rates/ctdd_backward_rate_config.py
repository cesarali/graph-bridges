from dataclasses import dataclass, field
from typing import List

@dataclass
class BackRateMLPConfig:
    name : str = 'BackRateMLP'

    # arquitecture variables
    ema_decay :float = 0.9999  # 0.9999
    time_embed_dim : int = 9
    hidden_layer : int = 200

    # reference process variables
    initial_dist : str = 'gaussian'
    rate_sigma : float = 6.0
    Q_sigma : float = 512.0
    time_exponential : float = 3.
    time_base :float = 1.0


@dataclass
class GaussianTargetRateImageX0PredEMAConfig:
    name :str  = 'GaussianTargetRateImageX0PredEMA'

    # arquitecture variables
    ema_decay : float = 0.9999  # 0.9999
    do_ema :bool = True
    fix_logistic :bool = False

    # reference process variables
    initial_dist :str = 'gaussian'
    rate_sigma :float = 6.0
    Q_sigma :float = 512.0
    time_exponential :float = 3.
    time_base :float = 1.0

@dataclass
class BackwardRateTemporalHollowTransformerConfig:
    name :str  = 'BackwardRateTemporalHollowTransformer'

    # arquitecture variables
    ema_decay : float = 0.9999  # 0.9999
    do_ema : bool = True

    num_scales :int = 4
    input_channels :int = 1
    scale_count_to_put_attn : int = 1
    dropout :float= 0.1
    skip_rescale :bool = True


    # reference process variables
    initial_dist :str = 'gaussian'
    rate_sigma :float = 6.0
    Q_sigma :float = 512.0
    time_exponential :float = 3.
    time_base :float = 1.0


all_backward_rates_configs = {"BackRateMLP":BackRateMLPConfig,
                              "GaussianTargetRateImageX0PredEMA":GaussianTargetRateImageX0PredEMAConfig,
                              "BackwardRateTemporalHollowTransformer":BackwardRateTemporalHollowTransformerConfig}