import os
import sys
from typing import List,Tuple
import torch
from matplotlib import pyplot as plt

def plot_histograms(marginal_histograms:Tuple[torch.Tensor],plots_path=None):
    """

    :param marginal_histograms: List[] marginal_0,marginal_generated_0,marginal_1,marginal_noising_1
    :param plots_path:
    :return:
    """
    marginal_0,marginal_generated_0,marginal_1,marginal_noising_1 = marginal_histograms
    marginal_0,marginal_generated_0,marginal_1,marginal_noising_1 = marginal_0.numpy(),marginal_generated_0.numpy(),marginal_1.numpy(),marginal_noising_1.numpy()

    fig, (ax1,ax2) = plt.subplots(ncols=2,nrows=1,figsize=(12,4))

    bin_edges = range(len(marginal_0))
    ax1.bar(bin_edges, marginal_generated_0, align='edge', width=1.0,alpha=0.2,label="generated_0")
    ax1.bar(bin_edges, marginal_0, align='edge', width=1.0,alpha=0.2,label="data ")

    ax1.set_title(r'Time 0')
    ax1.set_xlabel('Bins')
    ax1.set_ylabel('Counts')
    ax1.legend(loc="upper right")

    ax2.bar(bin_edges,marginal_noising_1 , align='edge', width=1.0,alpha=0.2,label="generated_0")
    ax2.bar(bin_edges, marginal_1, align='edge', width=1.0,alpha=0.2,label="data ")

    ax2.set_title(r'Time 1')
    ax2.set_xlabel('Bins')
    ax2.set_ylabel('Counts')
    ax2.legend(loc="upper right")

    ax2.set_ylim(0, 1.)
    ax1.set_ylim(0, 1.)

    if plots_path is None:
        plt.show()
    else:
        plt.savefig(plots_path)