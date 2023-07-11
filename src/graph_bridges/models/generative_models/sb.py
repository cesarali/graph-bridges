from graph_bridges.models.backward_rates.backward_rate import GaussianTargetRateImageX0PredEMA
from graph_bridges.models.schedulers.scheduling_utils import create_scheduler
from graph_bridges.models.backward_rates.backward_rate_utils import create_model
from graph_bridges.models.reference_process.reference_process_utils import create_reference
from graph_bridges.data.dataloaders_utils import create_dataloader
from graph_bridges.data.dataloaders import GraphSpinsDataLoader
from graph_bridges.models.losses.loss_utils import create_loss
from graph_bridges.models.losses.ctdd_losses import GenericAux
from graph_bridges.models.reference_process.ctdd_reference import ReferenceProcess
from graph_bridges.models.pipelines.sb.pipeline_sb import SBPipeline
from graph_bridges.models.schedulers.scheduling_sb import SBScheduler
from dataclasses import dataclass


@dataclass
class SB:
    """
    This class integrates all the objects requiered to train and generate data
    from a CTDD model

    """
    pipeline: SBPipeline=None
    data_dataloader: GraphSpinsDataLoader=None
    training_model: GaussianTargetRateImageX0PredEMA=None
    past_model: GaussianTargetRateImageX0PredEMA=None

    reference_process: ReferenceProcess=None
    loss: GenericAux=None
    scheduler: SBScheduler=None
    pipeline: SBPipeline=None

    def create_from_config(self,config,device):
        self.data_dataloader = create_dataloader(config, device)
        self.target_dataloader = create_dataloader(config, device,target=True)

        self.training_model = create_model(config, device)
        self.past_model = create_model(config, device)

        self.reference_process = create_reference(config, device)
        self.loss = create_loss(config, device)
        self.scheduler = SBScheduler(config, device)

        self.pipeline = SBPipeline(config,
                                   self.reference_process,
                                   self.data_dataloader,
                                   self.target_dataloader,
                                   self.scheduler)