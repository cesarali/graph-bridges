from dataclasses import dataclass

@dataclass
class SchrodingerBridgeBackwardRateConfig:
    name : str = 'SchrodingerBridgeBackwardRate'

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