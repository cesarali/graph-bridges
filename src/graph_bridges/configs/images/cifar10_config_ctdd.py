from dataclasses import asdict

from graph_bridges.configs.config_ctdd import CTDDConfig as GeneralCTDDConfig
from graph_bridges.data.image_dataloader_config import DiscreteCIFAR10Config
from graph_bridges.models.backward_rates.ctdd_backward_rate_config import GaussianTargetRateImageX0PredEMAConfig
from graph_bridges.models.temporal_networks.unets.unet_wrapper import UnetTauConfig

from dataclasses import dataclass

@dataclass
class CTDDConfig(GeneralCTDDConfig):

    config_path : str = ""

    # files, directories and naming ---------------------------------------------
    delete :bool = False
    experiment_name :str = 'cifar10'
    experiment_type :str = 'ctdd'
    experiment_indentifier :str  = 'testing'
    init_model_path = None

    # devices and parallelization ----------------------------------------------
    device = 'cpu'
    # device_paths = 'cpu' # not used
    distributed = False
    num_gpus = 0

    data:DiscreteCIFAR10Config = DiscreteCIFAR10Config()  # corresponds to the distributions at start time
    model:GaussianTargetRateImageX0PredEMAConfig = GaussianTargetRateImageX0PredEMAConfig()
    temp_network:UnetTauConfig =  UnetTauConfig(input_channels=3,ch_mult=[1, 2, 2, 2],data_min_max=[0,255])

    def align_configurations(self):
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

    config = CTDDConfig()
    pprint(asdict(config))