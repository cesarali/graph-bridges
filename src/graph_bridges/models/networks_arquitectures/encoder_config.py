from dataclasses import dataclass

@dataclass
class EncoderConfig:
    name:str = "Encoder"
    encoder_hidden_size:int = 50
    stochastic:bool = True
