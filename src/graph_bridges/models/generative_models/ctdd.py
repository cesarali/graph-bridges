from graph_bridges.models.backward_rates.backward_rate import GaussianTargetRateImageX0PredEMA
from graph_bridges.models.schedulers.scheduling_utils import create_scheduler
from graph_bridges.models.backward_rates.backward_rate_utils import create_model
from graph_bridges.models.reference_process.reference_process_utils import create_reference
from graph_bridges.data.dataloaders_utils import create_dataloader
from graph_bridges.models.schedulers.scheduling_ctdd import CTDDScheduler
from graph_bridges.data.dataloaders import GraphSpinsDataLoader
from graph_bridges.models.losses.loss_utils import create_loss
from graph_bridges.models.losses.ctdd_losses import GenericAux
from graph_bridges.models.pipelines.ctdd.pipeline_ctdd import CTDDPipeline
from graph_bridges.models.reference_process.ctdd_reference import ReferenceProcess

from dataclasses import dataclass


@dataclass
class CTDD:
    """
    This class integrates all the objects requiered to train and generate data
    from a CTDD model

    """
    data_dataloader: GraphSpinsDataLoader = None
    model: GaussianTargetRateImageX0PredEMA = None
    reference_process: ReferenceProcess = None
    loss: GenericAux = None
    scheduler: CTDDScheduler = None

    def create_from_config(self,config,device):
        self.data_dataloader = create_dataloader(config, device)
        self.target_dataloader = create_dataloader(config, device,target=True)
        self.model = create_model(config, device)
        self.reference_process = create_reference(config, device)
        self.loss = create_loss(config, device)
        self.scheduler = create_scheduler(config, device)

        self.pipeline = CTDDPipeline(config,
                                     self.reference_process,
                                     self.data_dataloader,
                                     self.target_dataloader,
                                     self.scheduler)