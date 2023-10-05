from dataclasses import dataclass

@dataclass
class DecoderConfig:
    name:str = "Decoder"
    decoder_hidden_size:int = 400
