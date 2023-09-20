from graph_bridges.models.temporal_networks.unets.unet_wrapper import UnetTauConfig
from graph_bridges.models.temporal_networks.convnets.autoencoder import ConvNetAutoencoderConfig
from graph_bridges.models.temporal_networks.transformers.temporal_hollow_transformers import TemporalHollowTransformerConfig



all_temp_nets_configs = {"TemporalHollowTransformer":TemporalHollowTransformerConfig,
                         "ConvNetAutoencoder":ConvNetAutoencoderConfig,
                         "UnetTau":UnetTauConfig}