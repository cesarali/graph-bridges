import torch

from graph_bridges.models.backward_rates.backward_rate import GaussianTargetRateImageX0PredEMA

from graph_bridges.data.dataloaders_utils import load_dataloader
from graph_bridges.models.backward_rates.backward_rate_utils import load_backward_rates

from graph_bridges.models.schedulers.scheduling_ctdd import CTDDScheduler
from graph_bridges.data.dataloaders import GraphSpinsDataLoader, DoucetTargetData
from graph_bridges.models.losses.ctdd_losses import GenericAux
from graph_bridges.models.pipelines.ctdd.pipeline_ctdd import CTDDPipeline
from graph_bridges.models.reference_process.ctdd_reference import GaussianTargetRate
from graph_bridges.configs.graphs.config_ctdd import CTDDConfig
import networkx as nx

from typing import List
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
    pipeline: CTDDPipeline = None

    def create_new_from_config(self, config:CTDDConfig, device):
        """

        :param config:
        :param device:
        :return:
        """
        config.initialize_new_experiment()

        self.config = config
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
            graphs_.append(nx.from_numpy_array(adj_matrices[graph_index].numpy()))
        return graphs_



