
from dataclasses import dataclass
from graph_bridges.configs.config_sb import get_sb_config_from_file
from pathlib import Path

import torch


from graph_bridges.data.dataloaders_utils import load_dataloader
from graph_bridges.data.graph_dataloaders import BridgeGraphDataLoaders

from graph_bridges.models.metrics.metrics_utils import read_metric
from graph_bridges.utils.test_utils import check_model_devices

from graph_bridges.configs.config_oops import OopsConfig
from graph_bridges.models.networks_arquitectures.rbf import RBM
from graph_bridges.models.pipelines.oops.pipeline_oops import OopsPipeline
from graph_bridges.models.networks_arquitectures.network_utils import load_model_network

import shutil
import re

@dataclass
class OOPS:
    """
    This class integrates all the objects requiered to train and generate data
    from a CTDD model

    """
    oops_config: OopsConfig = None

    dataloader : BridgeGraphDataLoaders = None
    model: RBM = None
    pipeline: OopsPipeline = None

    #metrics_registered = ["mse_histograms"]
    metrics_registered = []

    def create_new_from_config(self, config:OopsConfig, device):
        self.config = config
        self.config.initialize_new_experiment()
        self.dataloader = load_dataloader(config, type="data", device=device)
        self.model = load_model_network(config, device)
        self.pipeline = OopsPipeline(config,model=self.model,data=self.dataloader)

    def load_from_results_folder(self,
                                 experiment_name="graph",
                                 experiment_type="sb",
                                 experiment_indentifier="tutorial_sb_trainer",
                                 experiment_dir=None,
                                 checkpoint=None,
                                 any=False,
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
        results_ = None

        config_ready:OopsConfig = None

        """
        config_ready = get_config_from_file(experiment_name=experiment_name,
                                            experiment_type=experiment_type,
                                            experiment_indentifier=experiment_indentifier,
                                            results_dir=experiment_dir)
        """
        # LOADS RESULTS
        loaded_path = None
        if checkpoint is None:
            pass
        else:
            pass

        if loaded_path is None:
            print("Experiment Empty")
            return None

        if device is None:
            device = torch.device(config_ready.trainer.device)

        # SETS MODELS

        # JUST READs
        config_ready.align_configurations()
        self.set_classes_from_config(config_ready, device)
        self.config = config_ready

        #READ METRICS IF AVAILABLE
        all_metrics = {}
        for metric_string_identifier in self.metrics_registered:
            all_metrics.update(read_metric(self.config,
                                           metric_string_identifier,
                                           checkpoint=checkpoint))

        return results_,all_metrics,device

    def generate(self):
        """
        :param number_of_graphs:
        :return:
        """
        return None
