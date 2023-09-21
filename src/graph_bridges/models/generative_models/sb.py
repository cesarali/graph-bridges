from graph_bridges.models.backward_rates.ctdd_backward_rate import GaussianTargetRateImageX0PredEMA
from graph_bridges.models.reference_process.ctdd_reference import ReferenceProcess

from dataclasses import dataclass
from graph_bridges.models.pipelines.sb.pipeline_sb import SBPipeline
from graph_bridges.models.schedulers.scheduling_sb import SBScheduler
from graph_bridges.models.losses.estimators import BackwardRatioSteinEstimator
from graph_bridges.configs.graphs.graph_config_sb import SBConfig
from graph_bridges.configs.config_sb import get_sb_config_from_file
from pathlib import Path

import torch
import networkx as nx
from typing import List

from graph_bridges.data.graph_dataloaders import DoucetTargetData
from graph_bridges.data.dataloaders_utils import load_dataloader
from graph_bridges.data.graph_dataloaders import BridgeGraphDataLoaders
from graph_bridges.models.reference_process.ctdd_reference import GaussianTargetRate

from graph_bridges.models.backward_rates.backward_rate_utils import load_backward_rates
from graph_bridges.models.reference_process.reference_process_utils import load_reference
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

    def set_classes_from_config(self,config,device):
        self.data_dataloader = load_dataloader(config, type="data", device=device)
        self.target_dataloader = load_dataloader(config, type="target", device=device)

        self.reference_process = load_reference(config,device)


        self.backward_ratio_stein_estimator = BackwardRatioSteinEstimator(config, device)
        self.scheduler = SBScheduler(config, device)
        self.pipeline = SBPipeline(config,
                                   self.reference_process,
                                   self.data_dataloader,
                                   self.target_dataloader,
                                   self.scheduler)
        self.config = config

    def create_new_from_config(self, config:SBConfig, device):
        self.config = config
        self.config.initialize_new_experiment()
        self.set_classes_from_config(self.config,device)

        self.training_model = load_backward_rates(config, device)
        self.past_model = load_backward_rates(config, device)

    def load_from_results_folder(self,experiment_name="graph",
                                 experiment_type="sb",
                                 experiment_indentifier="tutorial_sb_trainer",
                                 sinkhorn_iteration_to_load=0,
                                 checkpoint=None,
                                 device=torch.device("cpu")):
        config_ready:SBConfig
        config_ready = get_sb_config_from_file(experiment_name=experiment_name,
                                               experiment_type=experiment_type,
                                               experiment_indentifier=experiment_indentifier)
        if checkpoint is None:
            best_model_to_load_path = Path(config_ready.experiment_files.best_model_path.format(sinkhorn_iteration_to_load))
            if best_model_to_load_path.exists():
                results_ = torch.load(best_model_to_load_path)
        else:
            check_point_to_load_path = Path(config_ready.experiment_files.best_model_path_checkpoint.format(checkpoint, sinkhorn_iteration_to_load))
            if check_point_to_load_path.exists():
                results_ = torch.load(check_point_to_load_path)

        config_ready.align_configurations()
        self.set_classes_from_config(config_ready, device)

        if sinkhorn_iteration_to_load == 0:
            self.training_model = results_['current_model']
            self.past_model = load_backward_rates(config=config_ready,device=device)
        else:
            self.training_model = results_['current_model']
            self.past_model = results_["past_model"]

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
