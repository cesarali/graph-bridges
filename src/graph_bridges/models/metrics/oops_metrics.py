import torch
import numpy as np
import networkx as nx

from graph_bridges.models.metrics import mmd
from graph_bridges.models.generative_models.oops import OOPS
from graph_bridges.models.metrics.data_metrics import SpinBernoulliMarginal
from graph_bridges.models.metrics.evaluation.stats import eval_graph_list


def graph_metrics_for_oops(sb,current_model,device):
    """

    :param sb:
    :param device:

    :return:  generated_graph_list,test_graph_list
    """
    # GET GRAPH FROM GENERATIVE MODEL
    remaining = sb.config.data.test_size
    generated_graph_list = []
    for spins_path in sb.pipeline.paths_iterator(current_model,
                                                 sinkhorn_iteration=1,
                                                 device=device,
                                                 train=False,
                                                 return_path=False):
        adj_matrices = sb.data_dataloader.transform_to_graph(spins_path)
        number_of_graphs = adj_matrices.shape[0]
        adj_matrices = adj_matrices.cpu().detach().numpy()
        for graph_index in range(number_of_graphs):
            graph_ = nx.from_numpy_array(adj_matrices[graph_index])
            generated_graph_list.append(graph_)
            remaining -=1
            if remaining <= 0:
                break

    # GET GRAPH FROM TEST DATASET
    test_graph_list = []
    for databatch in sb.data_dataloader.test():
        x = databatch[0]
        adj_matrices = sb.data_dataloader.transform_to_graph(x)
        number_of_graphs = adj_matrices.shape[0]
        adj_matrices = adj_matrices.detach().numpy()
        for graph_index in range(number_of_graphs):
            graph_ = nx.from_numpy_array(adj_matrices[graph_index])
            test_graph_list.append(graph_)

    results_ = eval_graph_list(generated_graph_list, test_graph_list)
    return results_

def marginal_histograms(generative_sample,test_sample):
    """

    :param oops:
    :param device:

    :return: backward_histogram,forward_histogram,forward_time

    """

    #================================
    # HISTOGRAMS OF DATA
    #================================

    marginal_histograms_data = test_sample.sum(axis=0).detach().cpu().numpy()
    marginal_histograms_sample = generative_sample.sum(axis=0).detach().cpu().numpy()

    #================================
    # HISTOGRAMS OF SAMPLE
    #================================

    mse = np.mean((marginal_histograms_data - marginal_histograms_sample)**2.)

    return mse

def kmmd(samples_0,sample_1):
    kmmd = mmd.MMD(mmd.exp_avg_hamming, False)
    opt_stat = kmmd.compute_mmd(samples_0.cpu(),sample_1.cpu())
    return opt_stat