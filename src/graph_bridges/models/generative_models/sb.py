from graph_bridges.models.backward_rates.backward_rate import GaussianTargetRateImageX0PredEMA
from graph_bridges.models.reference_process.reference_process_utils import create_reference
from graph_bridges.models.reference_process.ctdd_reference import ReferenceProcess
from graph_bridges.data.dataloaders import GraphSpinsDataLoader


from graph_bridges.models.pipelines.sb.pipeline_sb import SBPipeline
from graph_bridges.models.schedulers.scheduling_sb import SBScheduler
from dataclasses import dataclass
from graph_bridges.models.losses.estimators import BackwardRatioSteinEstimator
from graph_bridges.configs.graphs.config_sb import SBConfig
from pathlib import Path
import torch
import networkx as nx
from typing import List

from graph_bridges.data.dataloaders import DoucetTargetData
from graph_bridges.data.dataloaders_utils import load_dataloader
from graph_bridges.data.graph_dataloaders import BridgeGraphDataLoaders
from graph_bridges.models.reference_process.ctdd_reference import GaussianTargetRate
from graph_bridges.models.backward_rates.backward_rate_utils import load_backward_rates
from graph_bridges.utils.test_utils import check_model_devices

@dataclass
class SB:
    """
    This class integrates all the objects requiered to train and generate data
    from a CTDD model

    """
    data_dataloader: BridgeGraphDataLoaders = None
    target_dataloader: DoucetTargetData = None

    training_model: GaussianTargetRateImageX0PredEMA=None
    past_model: GaussianTargetRateImageX0PredEMA=None

    reference_process: ReferenceProcess=None
    backward_ratio_stein_estimator: BackwardRatioSteinEstimator=None
    scheduler: SBScheduler=None
    pipeline: SBPipeline=None
    config: SBConfig = None

    def create_new_from_config(self, config:SBConfig, device):
        self.config = config
        self.config.initialize_new_experiment()
        self.data_dataloader = load_dataloader(config, type="data", device=device)
        self.target_dataloader = load_dataloader(config, type="target", device=device)
        self.training_model = load_backward_rates(config, device)
        self.past_model = load_backward_rates(config, device)

        self.reference_process = GaussianTargetRate(config, device)
        self.backward_ratio_stein_estimator = BackwardRatioSteinEstimator(config, device)
        self.scheduler = SBScheduler(config, device)

        self.pipeline = SBPipeline(config,
                                   self.reference_process,
                                   self.data_dataloader,
                                   self.target_dataloader,
                                   self.scheduler)

        self.config = config

    def create_from_existing_config(self,config):
        if isinstance(config, SBConfig):
            if config.config_path == "":
                config.initialize_new_experiment()
            if not Path(config.config_path).exists():
                config.initialize_new_experiment()
        return None

    def generate_graphs(self,
                        number_of_graphs,
                        generating_model,
                        sinkhorn_iteration=0)->List[nx.Graph]:
        """
        :param number_of_graphs:
        :return:
        """
        if generating_model is None:
            generating_model = self.training_model

        try:
            device = check_model_devices(generating_model)
        except:
            device = generating_model.device

        ready = False
        graphs_ = []
        remaining_graphs = number_of_graphs
        for spins_path in self.pipeline.paths_iterator(generating_model,
                                                       sinkhorn_iteration=sinkhorn_iteration,
                                                       device=device,
                                                       train=True,
                                                       return_path=False):
            adj_matrices = self.data_dataloader.transform_to_graph(spins_path)
            number_of_graphs = adj_matrices.shape[0]
            for graph_index in range(number_of_graphs):
                graphs_.append(nx.from_numpy_array(adj_matrices[graph_index].cpu().numpy()))
                remaining_graphs -= 1
                if remaining_graphs < 1:
                    ready = True
                    break
            if ready:
                break
        return  graphs_
