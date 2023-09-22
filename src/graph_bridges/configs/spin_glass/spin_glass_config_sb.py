import os
from pprint import pprint
from typing import Union

from dataclasses import asdict
from dataclasses import dataclass
from graph_bridges.configs.config_sb import SBConfig as GeneralSBConfig
from graph_bridges.data.spin_glass_dataloaders_config import ParametrizedSpinGlassHamiltonianConfig
from graph_bridges.models.reference_process.reference_process_config import GlauberDynamicsConfig,GaussianTargetRateConfig
@dataclass
class SBConfig(GeneralSBConfig):

    # files, directories and naming ---------------------------------------------
    delete :bool = False
    experiment_type: str = 'sb'
    experiment_name :str = 'graph'
    experiment_indentifier :str  = 'testing'

    data: ParametrizedSpinGlassHamiltonianConfig = ParametrizedSpinGlassHamiltonianConfig()
    reference: Union[GlauberDynamicsConfig,GaussianTargetRateConfig] = GaussianTargetRateConfig()

    def align_configurations(self):
        from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig, BackwardRateTemporalHollowTransformerConfig
        from graph_bridges.models.backward_rates.ctdd_backward_rate_config import GaussianTargetRateImageX0PredEMAConfig
        from graph_bridges.models.temporal_networks.mlp.temporal_mlp import TemporalMLPConfig

        from graph_bridges.models.temporal_networks.unets.unet_wrapper import UnetTauConfig
        from graph_bridges.models.temporal_networks.convnets.autoencoder import ConvNetAutoencoderConfig
        from graph_bridges.models.temporal_networks.transformers.temporal_hollow_transformers import TemporalHollowTransformerConfig

        if isinstance(self.reference,GlauberDynamicsConfig):
            self.sampler.define_min_t_from_number_of_steps()

        self.data.as_spins = True
        if not isinstance(self.data, ParametrizedSpinGlassHamiltonianConfig):
            raise Exception("Data Not For the Specified Configuration")

        if isinstance(self.model,BackRateMLPConfig):
            self.data.as_image = False
            self.temp_network = TemporalMLPConfig()

        elif isinstance(self.model,GaussianTargetRateImageX0PredEMAConfig):
            if isinstance(self.temp_network,ConvNetAutoencoderConfig):
                #dataloaders for training
                self.data.as_image = True
                self.data.flatten_adjacency = False
            elif isinstance(self.temp_network, UnetTauConfig):
                raise Exception("Unet Network not implemented for Graphs (Yet)")

        elif isinstance(self.model, BackwardRateTemporalHollowTransformerConfig):
            self.data.as_image = False
            self.data.flatten_adjacency = True
            if not isinstance(self.temp_network,TemporalHollowTransformerConfig):
                self.temp_network = TemporalHollowTransformerConfig(input_vocab_size=2,
                                                                    output_vocab_size=2,
                                                                    max_seq_length=self.data.D)
            else:
                self.temp_network : TemporalHollowTransformerConfig
                self.temp_network.input_vocab_size = 2
                self.temp_network.output_vocab_size = 2
                self.temp_network.max_seq_length = self.data.D

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
