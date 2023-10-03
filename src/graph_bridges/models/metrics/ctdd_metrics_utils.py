#def log_metrics(sb:SB, current_model, past_to_train_model, sinkhorn_iteration, epoch, device, metrics_to_log=None):

import json
import torch
from graph_bridges.utils.plots.histograms_plots import plot_histograms
from graph_bridges.models.metrics.histograms_metrics import marginals_histograms_mse
from graph_bridges.models.temporal_networks.graphs_networks.graph_plots import plot_graphs_list2
from graph_bridges.models.metrics.ctdd_metrics import marginal_histograms_for_ctdd, graph_metrics_for_ctdd
from graph_bridges.models.generative_models.ctdd import CTDD

def log_metrics(ctdd:CTDD, number_of_steps, device, metrics_to_log=None):
    """
    After the training procedure is done, the model is updated

    :return:
    """
    config = ctdd.config
    if metrics_to_log is None:
        metrics_to_log = config.trainer.metrics

    # HISTOGRAMS
    metric_string_name = "histograms"
    if metric_string_name in metrics_to_log:
        histograms_plot_path_ = config.experiment_files.plot_path.format("histograms_{0}".format(number_of_steps))
        marginal_histograms = marginal_histograms_for_ctdd(ctdd, config, device)
        plot_histograms(marginal_histograms, plots_path=histograms_plot_path_)
        metric_string_name = "mse_histograms"
        if metric_string_name in config.trainer.metrics:
            mse_1, mse_0 = marginals_histograms_mse(marginal_histograms)
            mse_metric_path = config.experiment_files.metrics_file.format(
                metric_string_name + "_{0}".format(number_of_steps))
            with open(mse_metric_path, "w") as f:
                json.dump({"mse_histograms_1": mse_1.tolist(),
                           "mse_histograms_0": mse_0.tolist()}, f)

    else:
        metric_string_name = "mse_histograms"
        if metric_string_name in metrics_to_log:
            marginal_histograms = marginal_histograms_for_ctdd(ctdd, config, device)
            mse_1, mse_0 = marginals_histograms_mse(marginal_histograms)
            mse_metric_path = config.experiment_files.metrics_file.format(
                metric_string_name + "_{0}".format(number_of_steps))
            with open(mse_metric_path, "w") as f:
                json.dump({"mse_histograms_1": mse_1.tolist(),
                           "mse_histograms_0": mse_0.tolist()}, f)

    # METRICS
    metric_string_name = "graphs"
    if metric_string_name in metrics_to_log:
        graph_metrics_path_ = config.experiment_files.metrics_file.format(
            metric_string_name + "_{0}".format(number_of_steps))
        graph_metrics = graph_metrics_for_ctdd(ctdd, device)
        with open(graph_metrics_path_, "w") as f:
            json.dump(graph_metrics, f)

    # PLOTS
    metric_string_name = "graph_plot"
    if metric_string_name in metrics_to_log:
        graph_plot_path_ = config.experiment_files.graph_plot_path.format("generative_{0}".format(number_of_steps))
        generated_graphs = ctdd.generate_graphs(number_of_graphs=36)
        plot_graphs_list2(generated_graphs, title="Generated 0", save_dir=graph_plot_path_)