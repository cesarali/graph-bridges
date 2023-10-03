from dataclasses import asdict,dataclass

from graph_bridges.configs.config_ctdd import CTDDConfig as GeneralCTDDConfig
from graph_bridges.data.image_dataloader_config import NISTLoaderConfig
from graph_bridges.models.backward_rates.ctdd_backward_rate_config import GaussianTargetRateImageX0PredEMAConfig
from graph_bridges.models.temporal_networks.convnets.autoencoder import ConvNetAutoencoderConfig
from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig

@dataclass
class CTDDConfig(GeneralCTDDConfig):

    config_path : str = ""

    # files, directories and naming ---------------------------------------------
    delete :bool = False
    experiment_name :str = 'mnist'
    experiment_type :str = 'ctdd'
    experiment_indentifier :str  = 'testing'
    init_model_path = None

    # devices and parallelization ----------------------------------------------
    #device = 'cpu'
    # device_paths = 'cpu' # not used
    #distributed = False
    #num_gpus = 0

    data: NISTLoaderConfig = NISTLoaderConfig()  # corresponds to the distributions at start time
    model: GaussianTargetRateImageX0PredEMAConfig = GaussianTargetRateImageX0PredEMAConfig()
    temp_network: ConvNetAutoencoderConfig = ConvNetAutoencoderConfig()

    def align_configurations(self):
        from graph_bridges.models.backward_rates.ctdd_backward_rate_config import GaussianTargetRateImageX0PredEMAConfig
        from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig, BackwardRateTemporalHollowTransformerConfig

        from graph_bridges.models.temporal_networks.convnets.autoencoder import ConvNetAutoencoderConfig
        from graph_bridges.models.temporal_networks.transformers.temporal_hollow_transformers import TemporalHollowTransformerConfig
        from graph_bridges.models.temporal_networks.unets.unet_wrapper import UnetTauConfig
        from graph_bridges.models.temporal_networks.mlp.temporal_mlp import TemporalMLPConfig

        self.data.as_spins = False

        if isinstance(self.model,BackRateMLPConfig):
            if isinstance(self.temp_network,TemporalMLPConfig):
                pass
            else:
                self.temp_network = TemporalMLPConfig()

        elif isinstance(self.model,GaussianTargetRateImageX0PredEMAConfig):
            if isinstance(self.temp_network,ConvNetAutoencoderConfig):
                pass
            elif isinstance(self.temp_network,UnetTauConfig):
                raise Exception("Unet Network not compatible with MNIST")
            
        elif isinstance(self.model, BackwardRateTemporalHollowTransformerConfig):

            if not isinstance(self.temp_network,TemporalHollowTransformerConfig):
                self.temp_network = TemporalHollowTransformerConfig(input_vocab_size=2,
                                                                    output_vocab_size=2,
                                                                    max_seq_length=self.data.D)
            else:
                self.temp_network : TemporalHollowTransformerConfig
                self.temp_network.input_vocab_size = 2
                self.temp_network.output_vocab_size = 2
                self.temp_network.max_seq_length = self.data.D

        #dataloaders for training
        self.data.as_image = True
        self.data.as_spins = False
        self.data.doucet = True

        # data distributions matches at the end
        self.target.batch_size = self.data.batch_size

        # target
        self.target.S = self.data.S
        self.target.D = self.data.D
        self.target.C = self.data.C
        self.target.H = self.data.H
        self.target.W = self.data.W

        # model matches reference process
        self.reference.initial_dist = self.model.initial_dist
        self.reference.rate_sigma = self.model.rate_sigma
        self.reference.Q_sigma = self.model.Q_sigma
        self.reference.time_exponential = self.model.time_exponential
        self.reference.time_base = self.model.time_base


if __name__=="__main__":
    from pprint import pprint
    config = CTDDConfig()
    pprint(asdict(config))
