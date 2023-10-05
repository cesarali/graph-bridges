from typing import Union
from graph_bridges.configs.config_vae import VAEConfig
from graph_bridges.models.networks_arquitectures.encoder import Encoder
from graph_bridges.models.networks_arquitectures.decoder import Decoder

def load_encoder(config:VAEConfig):
    if config.encoder.name == "Encoder":
        encoder = Encoder(config)
    else:
        raise Exception("No Classifier")
    return encoder

def load_decoder(config:VAEConfig):
    if config.encoder.name == "Encoder":
        decoder = Decoder(config)
    else:
        raise Exception("No Classifier")
    return decoder

"""
def load_classifier(config:SSVAEConfig):
    if config.classifier.name == "Classifier":
        decoder = Classifier(config)
    else:
        raise Exception("No Classifier")
    return decoder
"""