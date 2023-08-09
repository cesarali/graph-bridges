import torch

from typing import Tuple,Union,List
from graph_bridges.configs.graphs.lobster.config_base import BridgeConfig

from graph_bridges.data.dataloaders import BridgeDataLoader
from graph_bridges.data.dataloaders_utils import create_dataloader
from graph_bridges.models.backward_rates.backward_rate import BackwardRate
from graph_bridges.models.backward_rates.backward_rate_utils import create_model


class SchrodingerBridge:
    """

    """
    model : BackwardRate
    data : BridgeDataLoader
    target : BridgeDataLoader
    config : BridgeConfig

    def __init__(self,config:BridgeConfig):
        self.config = config

    def initialize_experiment(self):
        """
        here we create all the requiered objects from the configuration file

        :return:
        """
        device = torch.device(config.device)
        self.model = create_model(config,device)
        self.data = create_dataloader(config,device,target=False)
        self.target = create_dataloader(config, device,target=True)

    def read_experiment(self):
        raise Exception("Not Implemented !")


def main(config):
    bridge = SchrodingerBridge(config)
    bridge.initialize_experiment()

    bridge.data.sample()


    return None

if __name__=="__main__":
    config = BridgeConfig()
    main(config)