import os
import sys
import json
import torch
import numpy as np
import networkx as nx

from pathlib import Path
from graph_bridges.models.generative_models.sb import SB
from graph_bridges.utils.plots.sb_plots import sinkhorn_plot
from graph_bridges.models.metrics.evaluation.stats import eval_graph_list
from graph_bridges.models.metrics.data_metrics import SpinBernoulliMarginal

from graph_bridges.utils.test_utils import check_model_devices
from graph_bridges.data.graph_dataloaders import BinaryTensorToSpinsTransform, SpinsToBinaryTensor


def graph_metrics_for_sb(sb,current_model,device):
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

def paths_marginal_histograms(sb:SB,
                              sinkhorn_iteration:int,
                              device:torch.device,
                              current_model,
                              past_to_train_model,
                              exact_backward=True,
                              train=True):
    """

    :param sb:
    :param sinkhorn_iteration:
    :param device:
    :param current_model:
    :param past_to_train_model:
    :param plot_path:

    :return: backward_histogram,forward_histogram,forward_time

    """
    total_number_of_steps = sb.config.sampler.num_steps + 1
    number_of_spins = sb.config.data.number_of_spins
    assert device == check_model_devices(current_model)

    expected_sample_size = sb.config.data.training_size if train else sb.config.data.test_size

    histogram_path_1 = torch.zeros(total_number_of_steps, number_of_spins,device=device)
    histogram_path_2 = torch.zeros(total_number_of_steps, number_of_spins,device=device)

    if sb.config.data.as_spins:
        spinsToBinaryTensor = SpinsToBinaryTensor()

    if exact_backward:
        how_much_1 = 0
        how_much_2 = 0
        for spins_path_1, times_1 in sb.pipeline.paths_iterator(past_to_train_model,
                                                                sinkhorn_iteration=sinkhorn_iteration,
                                                                return_path=True,
                                                                return_path_shape=True,
                                                                sample_from_reference_native=sb.config.sampler.sample_from_reference_native,
                                                                train=train,
                                                                device=device):
            end_of_path = spins_path_1[:, -1, :]
            spins_path_2, times_2 = sb.pipeline(current_model,
                                                sinkhorn_iteration + 1,
                                                device=device,
                                                initial_spins=end_of_path,
                                                return_path=True,
                                                train=train,
                                                sample_from_reference_native=sb.config.sampler.sample_from_reference_native,
                                                return_path_shape=True)
            how_much_2+= spins_path_2.shape[0]
            if sb.config.data.as_spins:
                binary_path_1 = spinsToBinaryTensor(spins_path_1)
                binary_path_2 = spinsToBinaryTensor(spins_path_2)
            else:
                binary_path_1 = spins_path_1
                binary_path_2 = spins_path_2

            current_sum_1 = binary_path_1.sum(axis=0)
            current_sum_2 = binary_path_2.sum(axis=0)

            histogram_path_1 += current_sum_1
            histogram_path_2 += current_sum_2

            how_much_1 += spins_path_1.shape[0]
    else:
        how_much_1 = 0
        for spins_path_1, times_1 in sb.pipeline.paths_iterator(past_to_train_model,
                                                                sinkhorn_iteration=sinkhorn_iteration,
                                                                return_path=True,
                                                                return_path_shape=True,
                                                                sample_from_reference_native=sb.config.sampler.sample_from_reference_native,
                                                                train=train,
                                                                device=device):
            if sb.config.data.as_spins:
                binary_path_1 = spinsToBinaryTensor(spins_path_1)
            else:
                binary_path_1 = spins_path_1
            current_sum_1 = binary_path_1.sum(axis=0)
            histogram_path_1 += current_sum_1
            how_much_1 += spins_path_1.shape[0]

        how_much_2 = 0
        for spins_path_2, times_2 in sb.pipeline.paths_iterator(current_model,
                                                                sinkhorn_iteration + 1,
                                                                device=device,
                                                                return_path=True,
                                                                train=train,
                                                                sample_from_reference_native=sb.config.sampler.sample_from_reference_native,
                                                                return_path_shape=True):
            how_much_2 += spins_path_2.shape[0]
            if sb.config.data.as_spins:
                binary_path_2 = spinsToBinaryTensor(spins_path_2)
            else:
                binary_path_2 = spins_path_2
            current_sum_2 = binary_path_2.sum(axis=0)
            histogram_path_2 += current_sum_2

    if sinkhorn_iteration % 2 == 0:
        backward_time = times_2[0]
        forward_time = times_1[0]
        backward_histogram = histogram_path_2
        forward_histogram = histogram_path_1
    else:
        backward_time = times_1[0]
        forward_time = times_2[0]
        forward_histogram = histogram_path_2
        backward_histogram = histogram_path_1

    print(f"how much 1 {how_much_1}")
    print(f"how much 3 {how_much_2}")
    backward_histogram = torch.flip(backward_histogram, [0])
    assert  how_much_1 == how_much_2 == expected_sample_size

    return backward_histogram,forward_histogram,forward_time


def marginal_paths_histograms_plots(sb:SB,
                                    sinkhorn_iteration:int,
                                    device:torch.device,
                                    current_model,
                                    past_to_train_model,
                                    plot_path:str,
                                    exact_backward=True,
                                    train=True):
    """

    :param sb:
    :return: marginal_0,marginal_1,backward_histogram,forward_histogram,forward_time
    """
    # CHECK DATA METRICS
    from graph_bridges.models.metrics.data_metrics import SpinBernoulliMarginal

    type_ = "train" if train else "test"
    marginal_0 = SpinBernoulliMarginal(spin_dataloader=sb.data_dataloader)(type=type_)
    marginal_1 = SpinBernoulliMarginal(spin_dataloader=sb.target_dataloader)(type=type_)

    backward_histogram,forward_histogram,forward_time = paths_marginal_histograms(sb,
                                                                                  sinkhorn_iteration,
                                                                                  device,
                                                                                  current_model,
                                                                                  past_to_train_model,
                                                                                  exact_backward,
                                                                                  train)

    state_legends = [str(i) for i in range(backward_histogram.shape[-1])]

    expected_sample_size = sb.config.data.training_size if train else sb.config.data.test_size
    marginal_0 = marginal_0.cpu()/expected_sample_size
    marginal_1 = marginal_1.cpu()/expected_sample_size
    backward_histogram = backward_histogram.cpu()/expected_sample_size
    forward_histogram = forward_histogram.cpu()/expected_sample_size

    sinkhorn_plot(sinkhorn_iteration,
                  marginal_0,
                  marginal_1,
                  backward_histogram=backward_histogram,
                  forward_histogram=forward_histogram,
                  time_=forward_time.cpu(),
                  states_legends=state_legends,
                  save_path=plot_path)

    return marginal_0,marginal_1,backward_histogram,forward_histogram,forward_time