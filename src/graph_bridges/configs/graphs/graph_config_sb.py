import os
from pprint import pprint
from dataclasses import asdict
from graph_bridges.configs.config_sb import SBConfig as GeneralSBConfig
from graph_bridges.models.reference_process.reference_process_config import GlauberDynamicsConfig

from dataclasses import dataclass

@dataclass
class SBConfig(GeneralSBConfig):

    # files, directories and naming ---------------------------------------------
    delete :bool = False
    experiment_type: str = 'sb'
    experiment_name :str = 'graph'
    experiment_indentifier :str  = 'testing'

    def align_configurations(self):
        from graph_bridges.models.backward_rates.sb_backward_rate_config import SchrodingerBridgeBackwardRateConfig

        from graph_bridges.models.temporal_networks.mlp.temporal_mlp import TemporalMLPConfig
        from graph_bridges.models.temporal_networks.unets.unet_wrapper import UnetTauConfig
        from graph_bridges.models.temporal_networks.convnets.autoencoder import ConvNetAutoencoderConfig
        from graph_bridges.models.temporal_networks.transformers.temporal_hollow_transformers import TemporalHollowTransformerConfig

        self.data.as_spins = True
        if isinstance(self.reference,GlauberDynamicsConfig):
            self.sampler.define_min_t_from_number_of_steps()

        if isinstance(self.model,SchrodingerBridgeBackwardRateConfig):
            if isinstance(self.temp_network,TemporalMLPConfig):
                self.data.as_image = False
                self.data.flatten_adjacency = True
            elif isinstance(self.temp_network,ConvNetAutoencoderConfig):
                #dataloaders for training
                self.data.as_image = True
                self.data.flatten_adjacency = False
            elif isinstance(self.temp_network, UnetTauConfig):
                raise Exception("Unet Network not implemented for Graphs (Yet)")
            elif isinstance(self.temp_network,TemporalHollowTransformerConfig):
                self.data.as_image = False
                self.data.flatten_adjacency = True
                self.temp_network : TemporalHollowTransformerConfig
                self.temp_network.input_vocab_size = 2
                self.temp_network.output_vocab_size = 2
                self.temp_network.max_seq_length = self.data.D
        else:
            raise Exception("Backward Rate Exclusive for Schrodinger")

        self.data.__post_init__()
        # data distributions matches at the end
        self.target.batch_size = self.data.batch_size

        # target
        self.target.S = self.data.S
        self.target.D = self.data.D
        self.target.C = self.data.C
        self.target.H = self.data.H
        self.target.W = self.data.W
        self.target.shape = self.data.shape
        self.target.shape_ = self.data.shape_

        # model matches reference process
        self.reference.initial_dist = self.model.initial_dist
        self.reference.rate_sigma = self.model.rate_sigma
        self.reference.Q_sigma = self.model.Q_sigma
        self.reference.time_exponential = self.model.time_exponential
        self.reference.time_base = self.model.time_base


if __name__=="__main__":
    config = SBConfig()
    pprint(config)
