from dataclasses import dataclass

@dataclass
class TemporalHollowTransformerConfig:
    temp_name:str = "TemporalHollowTransformer"
    num_layers: int = 4
    num_heads: int = 8
    hidden_dim: int = 256
    ff_hidden_dim: int = 512
    input_vocab_size: int = 10000
    output_vocab_size: int = 10000
    max_seq_length: int = 50
    do_time_embed: bool = True
    time_embed_dim: int = 128
    time_scale_factor: float = 1000
    fix_logistic :bool = False