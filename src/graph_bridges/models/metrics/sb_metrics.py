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

def graph_metrics_and_histograms(sb:SB,sinkhorn_iteration:int):
    """

    :param sb:
    :return:
    """
    # CHECK DATA METRICS
    data_stats_path = Path(sb.config.data_stats)
    if data_stats_path.exists():
        data_stats = json.load(open(data_stats_path, "rb"))
    else:
        from graph_bridges.models.metrics.data_metrics import SpinBernoulliMarginal

        bernoulli_marginal = SpinBernoulliMarginal(spin_dataloader=sb.data_dataloader)
        marginal_0 = bernoulli_marginal()
        bernoulli_marginal = SpinBernoulliMarginal(spin_dataloader=sb.target_dataloader)
        marginal_1 = bernoulli_marginal()

        histogram_path_0 = torch.zeros(total_number_of_steps, number_of_spins)
        histogram_path_1 = torch.zeros(total_number_of_steps, number_of_spins)

        for spins_path_1, times_1 in sb.pipeline.paths_iterator(past_to_train_model,
                                                                sinkhorn_iteration=sinkhorn_iteration,
                                                                return_path_shape=True):

            # end_of_path = spins_path_1[:, -1, :].unsqueeze(1).unsqueeze(1)
            end_of_path = spins_path_1[:, -1, :]
            spins_path_2, times_2 = sb.pipeline(current_model,
                                                sinkhorn_iteration + 1,
                                                device,
                                                end_of_path,
                                                return_path=True,
                                                return_path_shape=True)

            binary_0 = (spins_path_1 + 1.) * .5
            binary_1 = (spins_path_2 + 1.) * .5

            current_sum_0 = binary_0.sum(axis=0)
            current_sum_1 = binary_1.sum(axis=0)

            histogram_path_0 += current_sum_0
            histogram_path_1 += current_sum_1

        print(histogram_path_0[0])
        print(histogram_path_0[-1])

        print(histogram_path_1[0])
        print(histogram_path_1[-1])

        state_legends = [str(i) for i in range(histogram_path_0.shape[-1])]

        sinkhorn_plot(sinkhorn_iteration,
                      marginal_0,
                      marginal_1,
                      backward_histogram=histogram_path_1,
                      forward_histogram=histogram_path_0,
                      time_=times_1,
                      states_legends=state_legends)


if __name__=="__main__":
    print("Hey!")