from graph_bridges.models.networks.unets.unet_wrapper import UnetTauConfig
from graph_bridges.models.networks.convnets.autoencoder import ConvNetAutoencoderConfig
from graph_bridges.models.networks.transformers.temporal_hollow_transformers import TemporalHollowTransformerConfig



all_temp_nets_configs = {"TemporalHollowTransformer":TemporalHollowTransformerConfig,
                         "ConvNetAutoencoder":ConvNetAutoencoderConfig,
                         "UnetTau":UnetTauConfig}