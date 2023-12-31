from dataclasses import asdict,dataclass

from graph_bridges.configs.config_sb import SBConfig as GeneralSBConfig
from graph_bridges.data.image_dataloader_config import NISTLoaderConfig
from graph_bridges.models.backward_rates.ctdd_backward_rate_config import GaussianTargetRateImageX0PredEMAConfig
from graph_bridges.models.temporal_networks.convnets.autoencoder import ConvNetAutoencoderConfig
from graph_bridges.models.reference_process.reference_process_config import GlauberDynamicsConfig,GaussianTargetRateConfig

from graph_bridges.data.image_dataloader_config import NISTLoaderConfig
from graph_bridges.models.backward_rates.sb_backward_rate_config import SchrodingerBridgeBackwardRateConfig
from graph_bridges.models.temporal_networks.transformers.temporal_hollow_transformers import TemporalHollowTransformerConfig
from graph_bridges.models.temporal_networks.mlp.temporal_mlp import TemporalMLPConfig
from graph_bridges.models.temporal_networks.unets.unet_wrapper import UnetTauConfig

@dataclass
class SBConfig(GeneralSBConfig):

    config_path : str = ""

    # files, directories and naming ---------------------------------------------
    delete :bool = False
    experiment_name :str = 'mnist'
    experiment_type :str = 'sb'
    experiment_indentifier :str  = 'testing'
    init_model_path = None

    # devices and parallelization ----------------------------------------------
    device = 'cpu'
    distributed = False
    num_gpus = 0

    def align_configurations(self):
        #dataloaders for training
        from graph_bridges.models.losses.loss_configs import GradientEstimatorConfig, SteinSpinEstimatorConfig,RealFlipConfig

        if isinstance(self.flip_estimator, GradientEstimatorConfig):
            self.data.as_spins = False
        elif isinstance(self.flip_estimator, SteinSpinEstimatorConfig):
            self.data.as_spins = True
        elif isinstance(self.flip_estimator, RealFlipConfig):
            self.data.as_spins = True

        self.sampler.define_min_t_from_number_of_steps()

        if not isinstance(self.data, NISTLoaderConfig):
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

        # target
        self.target.S = self.data.S
        self.target.D = self.data.D
        self.target.C = self.data.C
        self.target.H = self.data.H
        self.target.W = self.data.W


if __name__=="__main__":
    from pprint import pprint
    config = SBConfig()
    pprint(asdict(config))
