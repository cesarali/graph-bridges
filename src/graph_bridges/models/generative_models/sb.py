from graph_bridges.models.backward_rates.backward_rate import GaussianTargetRateImageX0PredEMA
from graph_bridges.models.backward_rates.backward_rate_utils import create_model
from graph_bridges.models.reference_process.reference_process_utils import create_reference
from graph_bridges.data.dataloaders_utils import create_dataloader
from graph_bridges.data.dataloaders import GraphSpinsDataLoader
from graph_bridges.models.reference_process.ctdd_reference import ReferenceProcess
from graph_bridges.models.pipelines.sb.pipeline_sb import SBPipeline
from graph_bridges.models.schedulers.scheduling_sb import SBScheduler
from dataclasses import dataclass
from graph_bridges.models.losses.estimators import BackwardRatioSteinEstimator
from graph_bridges.configs.graphs.config_sb import BridgeConfig
from pathlib import Path

@dataclass
class SB:
    """
    This class integrates all the objects requiered to train and generate data
    from a CTDD model

    """
    pipeline: SBPipeline=None
    data_dataloader: GraphSpinsDataLoader=None
    starting_sinkhorn: GraphSpinsDataLoader=None

    training_model: GaussianTargetRateImageX0PredEMA=None
    past_model: GaussianTargetRateImageX0PredEMA=None

    reference_process: ReferenceProcess=None
    backward_ration_stein_estimator: BackwardRatioSteinEstimator=None
    scheduler: SBScheduler=None
    pipeline: SBPipeline=None
    config: BridgeConfig = None

    def __init__(self,config,device):
        self.create_from_config(config,device)

    def create_from_config(self,config:BridgeConfig,device):
        if isinstance(config,BridgeConfig):
            if config.config_path == "":
                config.initialize()
            if not Path(config.config_path).exists():
                config.initialize()


        self.data_dataloader = create_dataloader(config, device)
        self.target_dataloader = create_dataloader(config, device,target=True)

        self.training_model = create_model(config, device)
        self.past_model = create_model(config, device)

        self.reference_process = create_reference(config, device)
        self.backward_ration_stein_estimator = BackwardRatioSteinEstimator(config, device)
        self.scheduler = SBScheduler(config, device)

        self.pipeline = SBPipeline(config,
                                   self.reference_process,
                                   self.data_dataloader,
                                   self.target_dataloader,
                                   self.scheduler)
        self.config = config