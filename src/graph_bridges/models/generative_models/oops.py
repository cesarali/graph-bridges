
from dataclasses import dataclass
from graph_bridges.configs.config_sb import get_sb_config_from_file
from pathlib import Path

import torch


from graph_bridges.data.dataloaders_utils import load_dataloader
from graph_bridges.data.graph_dataloaders import BridgeGraphDataLoaders

from graph_bridges.models.metrics.metrics_utils import read_metric

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
    config: OopsConfig = None

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
        self.pipeline = OopsPipeline(config,model=self.model,data=self.dataloader,device=device)

    def load_from_results_folder(self,
                                 experiment_name="oops",
                                 experiment_type="mnist",
                                 experiment_indentifier="test",
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
        all_metrics = {}
        device = None

        config_ready = get_config_from_file(experiment_name=experiment_name,
                                            experiment_type=experiment_type,
                                            experiment_indentifier=experiment_indentifier,
                                            results_dir=experiment_dir)

        if config_ready is not None:
            # DEVICE
            if device is None:
                device = torch.device(config_ready.trainer.device)

            # LOADS RESULTS
            results_ = config_ready.experiment_files.load_results(checkpoint=checkpoint)

            # SETS MODELS
            if results_ is not None:
                self.config = config_ready

                self.model = results_["model"].to(device)
                self.dataloader = load_dataloader(self.config, type="data", device=device)
                self.model = load_model_network(self.config, device)
                self.pipeline = OopsPipeline(self.config,model=self.model,data=self.dataloader)

            # READ METRICS IF AVAILABLE
            """
            # JUST READs
            config_ready.align_configurations()
            self.set_classes_from_config(config_ready, device)
    
            #READ METRICS IF AVAILABLE
            for metric_string_identifier in self.metrics_registered:
                all_metrics.update(read_metric(self.config,
                                               metric_string_identifier,
                                               checkpoint=checkpoint))
            """
        return results_,all_metrics,device

    def sample_graphs(self):
        return None

