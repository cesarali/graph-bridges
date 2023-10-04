import os
import sys
import torch
from graph_bridges.models.backward_rates.ctdd_backward_rate import GaussianTargetRateImageX0PredEMA
from graph_bridges.models.metrics.metrics_utils import read_metric
from graph_bridges.data.dataloaders_utils import load_dataloader
from graph_bridges.models.backward_rates.backward_rate_utils import load_backward_rates

from graph_bridges.models.schedulers.scheduling_ctdd import CTDDScheduler
from graph_bridges.data.graph_dataloaders import DoucetTargetData
from graph_bridges.data.graph_dataloaders import BridgeGraphDataLoaders
from graph_bridges.models.losses.ctdd_losses import GenericAux
from graph_bridges.models.pipelines.ctdd.pipeline_ctdd import CTDDPipeline
from graph_bridges.models.reference_process.ctdd_reference import GaussianTargetRate
from graph_bridges.configs.graphs.graph_config_ctdd import CTDDConfig
import networkx as nx
from pathlib import Path
from typing import List
from graph_bridges.configs.config_ctdd import CTDDConfig
from dataclasses import dataclass


@dataclass
class CTDD:
    """
    This class integrates all the objects required to train and generate data

    from a CTDD model, it also provides the functionality to load the models

    from the experiment files.
    """
    data_dataloader: BridgeGraphDataLoaders = None
    target_dataloader: DoucetTargetData = None
    model: GaussianTargetRateImageX0PredEMA = None
    reference_process: GaussianTargetRate = None
    loss: GenericAux = None
    scheduler: CTDDScheduler = None
    pipeline: CTDDPipeline = None

    def create_new_from_config(self, config:CTDDConfig, device):
        """

        :param config:
        :param device:
        :return:
        """
        self.config = config
        self.config.initialize_new_experiment()
        self.model = load_backward_rates(config, device)
        self.set_classes_from_config(self.config,device)

    def set_classes_from_config(self,config,device):
        self.data_dataloader = load_dataloader(config, type="data", device=device)
        self.target_dataloader = load_dataloader(config, type="target", device=device)

        self.reference_process = GaussianTargetRate(config, device)
        self.loss = GenericAux(config,device)
        self.scheduler = CTDDScheduler(config,device)
        self.pipeline = CTDDPipeline(config,
                                     self.reference_process,
                                     self.data_dataloader,
                                     self.target_dataloader,
                                     self.scheduler)

    def load_from_results_folder(self,
                                 experiment_name=None,
                                 experiment_type=None,
                                 experiment_indentifier=None,
                                 results_dir=None,
                                 checkpoint=None,
                                 device=None):
        """

        :param experiment_name:
        :param experiment_type:
        :param experiment_indentifier:
        :param sinkhorn_iteration_to_load:
        :param checkpoint:
        :param device:

        :return: results,metrics,device
        """
        from graph_bridges.configs.utils import get_config_from_file

        config_ready:CTDDConfig
        config_ready = get_config_from_file(experiment_name=experiment_name,
                                            experiment_type=experiment_type,
                                            experiment_indentifier=experiment_indentifier,
                                            results_dir=results_dir)
        self.config = config_ready

        # LOADS RESULTS
        loaded_path = None
        if checkpoint is None:
            best_model_to_load_path = Path(self.config.experiment_files.best_model_path)
            if best_model_to_load_path.exists():
                results_ = torch.load(best_model_to_load_path)
                loaded_path = best_model_to_load_path
        else:
            check_point_to_load_path = Path(self.config.experiment_files.best_model_path_checkpoint.format(checkpoint))
            if check_point_to_load_path.exists():
                results_ = torch.load(check_point_to_load_path)
                loaded_path = check_point_to_load_path

        if loaded_path is None:
            print("Experiment Empty")
            return None


        if device is None:
            device = torch.device(self.config.trainer.device)

        # SETS MODELS
        self.model = results_['current_model'].to(device)

        # SETS ALL OTHER CLASSES FROM CONFIG AND START NEW EXPERIMENT IF REQUIERED
        self.config.align_configurations()
        self.set_classes_from_config(self.config, device)

        # READ METRICS IF AVAILABLE
        all_metrics = {}
        for metric_string_identifier in ["graphs","mse_histograms"]:
            all_metrics.update(read_metric(self.config, metric_string_identifier, checkpoint=checkpoint))

        return results_,all_metrics, device

    def generate_graphs(self,number_of_graphs)->List[nx.Graph]:
        """

        :param number_of_graphs:
        :return:
        """
        x = self.pipeline(self.model,number_of_graphs)
        adj_matrices = self.data_dataloader.transform_to_graph(x)
        graphs_ = []
        number_of_graphs = adj_matrices.shape[0]
        for graph_index in range(number_of_graphs):
            graphs_.append(nx.from_numpy_array(adj_matrices[graph_index].cpu().numpy()))
        return graphs_



if __name__=="__main__":
    from graph_bridges.models.metrics.ctdd_metrics import marginal_histograms_for_ctdd
    from graph_bridges.utils.plots.histograms_plots import plot_histograms
    from graph_bridges import results_path

    results_dir = os.path.join(results_path,"graph","ctdd","mlp_test_5_community")

    ctdd = CTDD()
    #ctdd.load_from_results_folder(experiment_indentifier="mlp_test_5_community",experiment_name="graph",experiment_type="ctdd")
    results,metrics,device = ctdd.load_from_results_folder(results_dir=results_dir)

    marginal_histograms = marginal_histograms_for_ctdd(ctdd, ctdd.config, device)
    plot_histograms(marginal_histograms)
    print(metrics)