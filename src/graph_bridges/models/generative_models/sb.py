from graph_bridges.models.reference_process.ctdd_reference import ReferenceProcess
from graph_bridges.models.backward_rates.sb_backward_rate_config import SchrodingerBridgeBackwardRateConfig
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

from graph_bridges.data.dataloaders_utils import load_dataloader
from graph_bridges.data.graph_dataloaders import DoucetTargetData
from graph_bridges.data.graph_dataloaders import BridgeGraphDataLoaders
from graph_bridges.models.reference_process.ctdd_reference import GaussianTargetRate

from graph_bridges.models.metrics.metrics_utils import read_metric

from graph_bridges.models.backward_rates.backward_rate_utils import load_backward_rates
from graph_bridges.models.reference_process.reference_process_utils import load_reference
from graph_bridges.utils.test_utils import check_model_devices
import shutil

@dataclass
class SB:
    """
    This class integrates all the objects requiered to train and generate data
    from a CTDD model

    """
    data_dataloader: BridgeGraphDataLoaders = None
    target_dataloader: DoucetTargetData = None

    training_model: SchrodingerBridgeBackwardRateConfig=None
    past_model: SchrodingerBridgeBackwardRateConfig=None

    reference_process: ReferenceProcess = None
    backward_ratio_estimator: BackwardRatioSteinEstimator= None
    scheduler: SBScheduler=None
    pipeline: SBPipeline=None
    config: SBConfig = None

    metrics_registered = ["graphs", "mse_histograms"]

    def set_classes_from_config(self,config,device):
        self.data_dataloader = load_dataloader(config, type="data", device=device)
        self.target_dataloader = load_dataloader(config, type="target", device=device)
        self.reference_process = load_reference(config,device)
        self.backward_ratio_estimator = BackwardRatioSteinEstimator(config, device)

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

    def load_from_results_folder(self,
                                 experiment_name="graph",
                                 experiment_type="sb",
                                 experiment_indentifier="tutorial_sb_trainer",
                                 results_dir=None,
                                 new_experiment=False,
                                 new_experiment_indentifier=None,
                                 sinkhorn_iteration_to_load=0,
                                 checkpoint=None,
                                 device=None):
        """

        :param experiment_name:
        :param experiment_type:
        :param experiment_indentifier:
        :param sinkhorn_iteration_to_load:
        :param checkpoint:
        :param device:
        :return: results_,all_metrics,device
        """
        from graph_bridges.configs.utils import get_config_from_file

        config_ready:SBConfig
        config_ready = get_config_from_file(experiment_name=experiment_name,
                                            experiment_type=experiment_type,
                                            experiment_indentifier=experiment_indentifier,
                                            results_dir=results_dir)

        # LOADS RESULTS
        loaded_path = None
        if checkpoint is None:
            best_model_to_load_path = Path(config_ready.experiment_files.best_model_path.format(sinkhorn_iteration_to_load))
            if best_model_to_load_path.exists():
                results_ = torch.load(best_model_to_load_path)
                loaded_path = best_model_to_load_path
        else:
            check_point_to_load_path = Path(config_ready.experiment_files.best_model_path_checkpoint.format(checkpoint, sinkhorn_iteration_to_load))
            if check_point_to_load_path.exists():
                results_ = torch.load(check_point_to_load_path)
                loaded_path = check_point_to_load_path

        if loaded_path is None:
            print("Experiment Empty")
            return None


        if device is None:
            device = torch.device(config_ready.trainer.device)

        # SETS MODELS
        if sinkhorn_iteration_to_load == 0:
            self.training_model = results_['current_model'].to(device)
            self.past_model = load_backward_rates(config=config_ready,device=device)
        else:
            self.training_model = results_['current_model'].to(device)
            self.past_model = results_["past_model"].to(device)

        # SETS ALL OTHER CLASSES FROM CONFIG AND START NEW EXPERIMENT IF REQUIERED
        if new_experiment:
            # Creates a folder
            config_ready.experiment_indentifier = new_experiment_indentifier
            config_ready.__post_init__()
            config_ready.initialize_new_experiment()

            if checkpoint is None:
                best_model_to_load_path = Path(config_ready.experiment_files.best_model_path.format(sinkhorn_iteration_to_load))
                to_copy_file = best_model_to_load_path
            else:
                check_point_to_load_path = Path(config_ready.experiment_files.best_model_path_checkpoint.format(checkpoint,sinkhorn_iteration_to_load))
                to_copy_file = check_point_to_load_path

            shutil.copy2(loaded_path, to_copy_file)
            config_ready.align_configurations()
            self.set_classes_from_config(config_ready, device)

        # JUST READs
        else:
            config_ready.align_configurations()
            self.set_classes_from_config(config_ready, device)

        all_metrics = {}
        for metric_string_identifier in self.metrics_registered:
            all_metrics.update(read_metric(self.config,
                                           metric_string_identifier,
                                           sinkhorn_iteration=sinkhorn_iteration_to_load,
                                           checkpoint=checkpoint))

        return results_,all_metrics,device

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

        return graphs_
