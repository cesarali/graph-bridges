from dataclasses import dataclass

@dataclass
class SchrodingerBridgeBackwardRateConfig:
    name : str = 'SchrodingerBridgeBackwardRate'

    # arquitecture variables
    do_ema:bool = True
    ema_decay :float = 0.9999  # 0.9999
    time_embed_dim : int = 9
    hidden_layer : int = 200