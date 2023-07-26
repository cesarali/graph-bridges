import torch

from graph_bridges.models.backward_rates.backward_rate import GaussianTargetRateImageX0PredEMA

from graph_bridges.data import load_dataloader
from graph_bridges.models.backward_rates import load_backward_rates

from graph_bridges.models.schedulers.scheduling_ctdd import CTDDScheduler
from graph_bridges.data.dataloaders import GraphSpinsDataLoader, DoucetTargetData
from graph_bridges.models.losses.ctdd_losses import GenericAux
from graph_bridges.models.pipelines.ctdd.pipeline_ctdd import CTDDPipeline
from graph_bridges.models.reference_process.ctdd_reference import GaussianTargetRate
from graph_bridges.configs.graphs.config_ctdd import CTDDConfig

from dataclasses import dataclass, asdict
from pprint import pprint

@dataclass
class CTDD:
    """
    This class integrates all the objects requiered to train and generate data
    from a CTDD model

    """
    data_dataloader: GraphSpinsDataLoader = None
    target_dataloader: DoucetTargetData = None
    model: GaussianTargetRateImageX0PredEMA = None
    reference_process: GaussianTargetRate = None
    loss: GenericAux = None
    scheduler: CTDDScheduler = None

    def create_from_config(self,config:CTDDConfig,device):
        """

        :param config:
        :param device:
        :return:
        """
        config.initialize_new_experiment()

        self.data_dataloader = load_dataloader(config, type="data", device=device)
        self.target_dataloader = load_dataloader(config, type="target", device=device)
        self.model = load_backward_rates(config, device)

        self.reference_process = GaussianTargetRate(config, device)
        self.loss = GenericAux(config,device)
        self.scheduler = CTDDScheduler(config,device)

        self.pipeline = CTDDPipeline(config,
                                     self.reference_process,
                                     self.data_dataloader,
                                     self.target_dataloader,
                                     self.scheduler)




