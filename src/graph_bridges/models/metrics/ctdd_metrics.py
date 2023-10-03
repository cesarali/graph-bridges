import os
import sys
import json
import torch
import numpy as np
import networkx as nx
from pathlib import Path
from dataclasses import dataclass
from typing import Union,Tuple,List
from graph_bridges.configs.config_ctdd import CTDDConfig
from graph_bridges.utils.test_utils import check_model_devices
from graph_bridges.models.metrics.evaluation.stats import eval_graph_list
from graph_bridges.models.metrics.data_metrics import SpinBernoulliMarginal



def marginals_histograms_mse(all_marginal_histograms)->Tuple[np.array,np.array]:
    """
    simply calculates the mse from the marginal graph histograms

    Returns
    -------
    mse_1,mse_0
    """
    marginal_0, marginal_generated_0, marginal_1, marginal_noising_1 = all_marginal_histograms
    if isinstance(marginal_0,torch.Tensor):
        marginal_0 = marginal_0.numpy()
    if isinstance(marginal_generated_0,torch.Tensor):
        marginal_generated_0 = marginal_generated_0.numpy()
    if isinstance(marginal_1,torch.Tensor):
        marginal_1 = marginal_1.numpy()
    if isinstance(marginal_noising_1,torch.Tensor):
        marginal_noising_1 = marginal_noising_1.numpy()

    mse_1 = np.mean((marginal_1 - marginal_noising_1)**2.)
    mse_0 = np.mean((marginal_0 - marginal_generated_0)**2.)

    return mse_1,mse_0

def marginal_histograms_for_ctdd(ctdd,config,device):
    """

    :param ctdd:
    :param config:
    :param device:

    :return: marginal_0,marginal_generated_0,marginal_1,marginal_noising_1
    """

    if config.data.name in ["NISTLoader","mnist"]:
        data_loader_ = ctdd.data_dataloader.test()
        type = "test"
        try:
            size_ = int(config.data.test_size)
        except:
            size_ = config.data.total_data_size - int(config.data.total_data_size * config.data.training_proportion)
    else:
        type = "train"
        data_loader_ = ctdd.data_dataloader.train()
        try:
            size_ = int(config.data.training_size)
        except:
            size_ = int(config.data.total_data_size * config.data.training_proportion)

    # marginals from real data
    marginal_0 = SpinBernoulliMarginal(spin_dataloader=ctdd.data_dataloader)(type=type)
    marginal_1 = SpinBernoulliMarginal(spin_dataloader=ctdd.target_dataloader)(type=type)

    # marginals from generative models and noising
    marginal_generated_0 = torch.zeros(config.data.number_of_spins,device=device)
    marginal_noising_1 = torch.zeros(config.data.number_of_spins,device=device)

    batch_size = config.data.batch_size

    current_index = 0
    while current_index < size_:
        remaining = min(size_ - current_index, batch_size)
        x = ctdd.pipeline(ctdd.model, remaining,device=device)
        marginal_generated_0 += x.sum(axis=0)        # Your processing code here
        current_index += remaining

    # Sample a random timestep for each image
    how_much_ = 0
    for batchdata in data_loader_:
        x_adj = batchdata[0]
        ts = torch.ones(batch_size)
        x_adj = x_adj.to(device)
        ts = ts.to(device)
        how_much_ += x_adj.shape[0]

        x_t, x_tilde, qt0, rate = ctdd.scheduler.add_noise(x_adj,
                                                           ctdd.reference_process,
                                                           ts,
                                                           device,
                                                           return_dict=False)
        marginal_noising_1 += x_t.sum(axis=0)


    marginal_0 = marginal_0/size_
    marginal_generated_0 = marginal_generated_0.cpu()/size_
    marginal_1 = marginal_1/size_
    marginal_noising_1 = marginal_noising_1.cpu()/size_

    return marginal_0,marginal_generated_0,marginal_1,marginal_noising_1

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