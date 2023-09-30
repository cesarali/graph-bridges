import os
from pprint import pprint
from typing import Union

from dataclasses import asdict
from dataclasses import dataclass
from graph_bridges.configs.config_sb import SBConfig as GeneralSBConfig
from graph_bridges.data.spin_glass_dataloaders_config import ParametrizedSpinGlassHamiltonianConfig
from graph_bridges.models.reference_process.reference_process_config import GlauberDynamicsConfig,GaussianTargetRateConfig
from graph_bridges.models.backward_rates.sb_backward_rate_config import SchrodingerBridgeBackwardRateConfig

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
        from graph_bridges.models.losses.loss_configs import GradientEstimatorConfig,SteinSpinEstimatorConfig,RealFlipConfig

        if isinstance(self.flip_estimator,GradientEstimatorConfig):
            self.data.as_spins = False
        elif isinstance(self.flip_estimator,SteinSpinEstimatorConfig):
            self.data.as_spins = True
        elif isinstance(self.flip_estimator, RealFlipConfig):
            self.data.as_spins = True

        self.sampler.define_min_t_from_number_of_steps()

        if not isinstance(self.data, ParametrizedSpinGlassHamiltonianConfig):
            raise Exception("Data Not For the Specified Configuration")

        if isinstance(self.model,SchrodingerBridgeBackwardRateConfig):
            if isinstance(self.temp_network,TemporalMLPConfig):
                self.data.as_image = False
            elif isinstance(self.temp_network,ConvNetAutoencoderConfig):
                self.data.as_image = True
            elif isinstance(self.temp_network, UnetTauConfig):
                raise Exception("Unet Network not implemented for Graphs (Yet)")
            elif isinstance(self.temp_network,TemporalHollowTransformerConfig):
                self.data.as_image = False
                self.temp_network : TemporalHollowTransformerConfig
                self.temp_network.input_vocab_size = 2
                self.temp_network.output_vocab_size = 2
                self.temp_network.max_seq_length = self.data.D
        else:
            raise Exception("Backward Rate Exclusive for Schrodinger")
        self.data.__post_init__()

        # data distributions matches at the end
        self.target.batch_size = self.data.batch_size


if __name__=="__main__":
    config = SBConfig()
    pprint(config)
