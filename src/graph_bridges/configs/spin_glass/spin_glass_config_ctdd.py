import os
import sys
from pprint import pprint
from dataclasses import asdict,dataclass
from graph_bridges.configs.config_ctdd import CTDDConfig as GeneralCTDDConfig
from graph_bridges.data.spin_glass_dataloaders_config import ParametrizedSpinGlassHamiltonianConfig

@dataclass
class CTDDConfig(GeneralCTDDConfig):
    """

    """
    # files, directories and naming ---------------------------------------------
    delete :bool = False
    experiment_type: str = 'ctdd'
    experiment_name :str = 'graph'
    experiment_indentifier :str  = 'testing'

    data:ParametrizedSpinGlassHamiltonianConfig = ParametrizedSpinGlassHamiltonianConfig()

    def align_configurations(self):
        from graph_bridges.models.backward_rates.ctdd_backward_rate_config import GaussianTargetRateImageX0PredEMAConfig
        from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig, BackwardRateTemporalHollowTransformerConfig


        from graph_bridges.models.temporal_networks.unets.unet_wrapper import UnetTauConfig
        from graph_bridges.models.temporal_networks.mlp.temporal_mlp import TemporalMLPConfig
        from graph_bridges.models.temporal_networks.convnets.autoencoder import ConvNetAutoencoderConfig
        from graph_bridges.models.temporal_networks.transformers.temporal_hollow_transformers import TemporalHollowTransformerConfig

        self.data.as_spins = False
        self.data.doucet = True

        #----------------------------------------------------------------------------------------
        # HERE WE PREPARE THE DATA TRANSFORMATIONS TO PIN THE TEMPORAL NETWORK ARCHITECTURE
        #----------------------------------------------------------------------------------------

        if isinstance(self.model,BackRateMLPConfig):
            if isinstance(self.temp_network,TemporalMLPConfig):
                pass
            else:
                self.temp_network = TemporalMLPConfig()
            self.data.as_image = False

        elif isinstance(self.model,GaussianTargetRateImageX0PredEMAConfig):
            if isinstance(self.temp_network,ConvNetAutoencoderConfig):
                #dataloaders for training
                self.data.as_image = True

            elif isinstance(self.temp_network, UnetTauConfig):
                raise Exception("Unet Network not implemented for Graphs (Yet)")

        elif isinstance(self.model, BackwardRateTemporalHollowTransformerConfig):
            self.data.as_image = False

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

        # model matches reference process
        self.reference.initial_dist = self.model.initial_dist
        self.reference.rate_sigma = self.model.rate_sigma
        self.reference.Q_sigma = self.model.Q_sigma
        self.reference.time_exponential = self.model.time_exponential
        self.reference.time_base = self.model.time_base


if __name__=="__main__":
    config = CTDDConfig()
    pprint(asdict(config.data))
