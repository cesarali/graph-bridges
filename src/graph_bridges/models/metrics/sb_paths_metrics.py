import os
import sys
import torch
from graph_bridges.models.generative_models.sb import SB
from graph_bridges.utils.plots.sb_plots import sinkhorn_plot
from graph_bridges.utils.test_utils import check_model_devices
from graph_bridges.models.spin_glass.spin_states_statistics import spin_states_stats
from graph_bridges.data.transforms import SpinsToBinaryTensor,BinaryTensorToSpinsTransform

def paths_states_histograms(sb:SB,
                            sinkhorn_iteration:int,
                            device:torch.device,
                            current_model,
                            past_to_train_model):
    """

    :param sb:
    :param sinkhorn_iteration:
    :param device:
    :param current_model:
    :param past_to_train_model:
    :param plot_path:

    :return: backward_histogram,forward_histogram,forward_time

    """
    stats_ = spin_states_stats(sb.config.data.number_of_spins)
    number_of_total_states = stats_.number_of_total_states

    total_number_of_steps = sb.config.sampler.num_steps + 1
    number_of_spins = sb.config.data.number_of_spins
    assert device == check_model_devices(current_model)

    histogram_path_1 = torch.zeros(total_number_of_steps, number_of_total_states,device=device)
    histogram_path_2 = torch.zeros(total_number_of_steps, number_of_total_states,device=device)

    for spins_path_1, times_1 in sb.pipeline.paths_iterator(past_to_train_model,
                                                            sinkhorn_iteration=sinkhorn_iteration,
                                                            return_path=True,
                                                            return_path_shape=True,
                                                            device=device):
        end_of_path = spins_path_1[:, -1, :]
        spins_path_2, times_2 = sb.pipeline(current_model,
                                            sinkhorn_iteration + 1,
                                            device=device,
                                            initial_spins=end_of_path,
                                            return_path=True,
                                            return_path_shape=True)

        current_sum_1 = stats_.counts_states_in_paths(spins_path_1.cpu()).to(device)
        current_sum_2 = stats_.counts_states_in_paths(spins_path_2.cpu()).to(device)

        histogram_path_1 += current_sum_1
        histogram_path_2 += current_sum_2

    if sinkhorn_iteration % 2 == 0:
        backward_time = times_2[0]
        forward_time = times_1[0]
        backward_histogram = histogram_path_2
        forward_histogram = histogram_path_1
    else:
        backward_time = times_1[0]
        forward_time = times_2[0]
        backward_histogram = histogram_path_2
        forward_histogram = histogram_path_1

    backward_histogram = torch.flip(backward_histogram, [0])

    return backward_histogram,forward_histogram,forward_time

def states_paths_histograms_plots(sb:SB,
                                  sinkhorn_iteration:int,
                                  device:torch.device,
                                  current_model,
                                  past_to_train_model,
                                  plot_path:str):
    """

    :param sb:
    :return: marginal_0,marginal_1,backward_histogram,forward_histogram,forward_time
    """
    # CHECK DATA METRICS
    stats_ = spin_states_stats(sb.config.data.number_of_spins)
    number_of_total_states = stats_.number_of_total_states

    marginal_0 =  torch.zeros(number_of_total_states)
    marginal_1 = torch.zeros(number_of_total_states)

    for databatch in sb.data_dataloader.train():
        spin_states = databatch[0]
        marginal_0 += stats_.counts_for_different_states(spin_states)

    for databatch in sb.data_dataloader.train():
        spin_states = databatch[0]
        marginal_1 += stats_.counts_for_different_states(spin_states)

    backward_histogram,forward_histogram,forward_time = paths_states_histograms(sb,
                                                                                sinkhorn_iteration,
                                                                                device,
                                                                                current_model,
                                                                                past_to_train_model)
    state_legends = [str(state) for state in stats_.all_states_in_order]

    sinkhorn_plot(sinkhorn_iteration,
                  marginal_0.cpu(),
                  marginal_1.cpu(),
                  backward_histogram=backward_histogram.cpu(),
                  forward_histogram=forward_histogram.cpu(),
                  time_=forward_time.cpu(),
                  states_legends=state_legends,
                  save_path=plot_path)

    return marginal_0,marginal_1,backward_histogram,forward_histogram,forward_time
