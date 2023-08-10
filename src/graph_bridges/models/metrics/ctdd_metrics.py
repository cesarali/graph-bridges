import os
import sys
import torch
import numpy as np
import networkx as nx
from graph_bridges.models.metrics.evaluation.stats import eval_graph_list
from graph_bridges.models.metrics.data_metrics import SpinBernoulliMarginal
from graph_bridges.utils.test_utils import check_model_devices

def marginal_histograms_for_ctdd(ctdd,config,device):
    """

    :param ctdd:
    :param config:
    :param device:

    :return: marginal_0,marginal_generated_0,marginal_1,marginal_noising_1
    """
    # marginals from real data
    marginal_0 = SpinBernoulliMarginal(spin_dataloader=ctdd.data_dataloader)()
    marginal_1 = SpinBernoulliMarginal(spin_dataloader=ctdd.target_dataloader)()

    # marginals from generative models and noising
    marginal_generated_0 = torch.zeros(config.data.number_of_spins,device=device)
    marginal_noising_1 = torch.zeros(config.data.number_of_spins,device=device)

    training_size = int(config.data.total_data_size * config.data.training_proportion)
    batch_size = config.data.batch_size
    number_of_batches = int(training_size / batch_size)

    for batch_index in range(number_of_batches):
        x = ctdd.pipeline(ctdd.model, batch_size,device=device)
        marginal_generated_0 += x.sum(axis=0)

    # Sample a random timestep for each image
    for batchdata in ctdd.data_dataloader.train():
        x_adj = batchdata[0]
        ts = torch.ones(batch_size)
        x_adj = x_adj.to(device)
        ts = ts.to(device)

        x_t, x_tilde, qt0, rate = ctdd.scheduler.add_noise(x_adj,
                                                           ctdd.reference_process,
                                                           ts,
                                                           device,
                                                           return_dict=False)
        marginal_noising_1 += x_t.sum(axis=0)

    return marginal_0,marginal_generated_0.cpu(),marginal_1,marginal_noising_1.cpu()

def graph_metrics_for_ctdd(ctdd,device):
    x = ctdd.pipeline(ctdd.model, ctdd.data_dataloader.test_data_size,device = device)
    adj_matrices = ctdd.data_dataloader.transform_to_graph(x)

    # GET GRAPH FROM GENERATIVE MODEL
    generated_graph_list = []
    number_of_graphs = adj_matrices.shape[0]
    adj_matrices = adj_matrices.detach().cpu().numpy()
    for graph_index in range(number_of_graphs):
        graph_ = nx.from_numpy_array(adj_matrices[graph_index])
        generated_graph_list.append(graph_)

    # GET GRAPH FROM TEST DATASET
    test_graph_list = []
    for databatch in ctdd.data_dataloader.test():
        x = databatch[0]
        adj_matrices = ctdd.data_dataloader.transform_to_graph(x)
        number_of_graphs = adj_matrices.shape[0]
        adj_matrices = adj_matrices.detach().cpu().numpy()
        for graph_index in range(number_of_graphs):
            graph_ = nx.from_numpy_array(adj_matrices[graph_index])
            test_graph_list.append(graph_)

    results_ = eval_graph_list(generated_graph_list, test_graph_list)
    return results_