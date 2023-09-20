
from .unets.unet_wrapper import UnetTau
from .transformers.temporal_hollow_transformers import TemporalHollowTransformer
from .convnets.autoencoder import ConvNetAutoencoder

# From https://github.com/yang-song/score_sde_pytorch/ which is from
# https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py

def load_temp_network(config, device):
  if config.temp_network.temp_name == "UnetTau":
    image_network = UnetTau(config,device)
  elif config.temp_network.temp_name == "TemporalHollowTransformer":
    image_network = TemporalHollowTransformer(config,device)
  elif config.temp_network.temp_name == "ConvNetAutoencoder":
    image_network = ConvNetAutoencoder(config, device)
  else:
    raise Exception("No UnetNetwork")
  return image_network
