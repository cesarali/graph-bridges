from dataclasses import asdict,dataclass

from graph_bridges.configs.config_sb import SBConfig as GeneralSBConfig
from graph_bridges.data.image_dataloader_config import NISTLoaderConfig
from graph_bridges.models.backward_rates.ctdd_backward_rate_config import GaussianTargetRateImageX0PredEMAConfig
from graph_bridges.models.temporal_networks.convnets.autoencoder import ConvNetAutoencoderConfig
from graph_bridges.models.reference_process.reference_process_config import GlauberDynamicsConfig,GaussianTargetRateConfig

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
    # device_paths = 'cpu' # not used
    distributed = False
    num_gpus = 0

    def __post_init__(self):
        self.data = NISTLoaderConfig()  # corresponds to the distributions at start time
        self.model = GaussianTargetRateImageX0PredEMAConfig()
        self.temp_network = ConvNetAutoencoderConfig()

    def align_configurations(self):
        if isinstance(self.reference,GlauberDynamicsConfig):
            self.sampler.define_min_t_from_number_of_steps()

        #dataloaders for training
        self.data.as_image = True
        self.data.as_spins = False

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
    config = SBConfig()
    pprint(asdict(config))
