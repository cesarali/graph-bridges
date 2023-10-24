import os
import json
import torch
from graph_bridges.models.generative_models.oops import OOPS
from graph_bridges.models.metrics.oops_metrics import marginal_histograms
from graph_bridges.models.metrics.oops_metrics import kmmd

def store_metrics(config,all_metrics,new_metrics,metric_string_name,epoch,where_to_log=None):
    if where_to_log is None:
        mse_metric_path = config.experiment_files.metrics_file.format(metric_string_name + "_{0}_".format(epoch))
    else:
        mse_metric_path = where_to_log[metric_string_name]

    all_metrics.update(new_metrics)
    with open(mse_metric_path, "w") as f:
        json.dump(new_metrics, f)
    return all_metrics

def log_metrics(oops:OOPS, epoch, device, metrics_to_log=None, where_to_log=None, writer=None):
    """
    After the training procedure is done, the model is updated

    :return:
    """
    all_metrics = {}

    config = oops.config
    if metrics_to_log is None:
        metrics_to_log = config.trainer.metrics

    #OBTAIN SAMPLES
    test_sample = torch.vstack([databatch[0] for databatch in oops.dataloader.test()])
    generative_sample = oops.pipeline(sample_size=test_sample.shape[0])

    # HISTOGRAMS
    metric_string_name = "mse_histograms"
    if metric_string_name in metrics_to_log:
        mse_marginal_histograms = marginal_histograms(generative_sample,test_sample)
        mse_metrics = {"mse_marginal_histograms": mse_marginal_histograms.tolist()}
        all_metrics = store_metrics(config, all_metrics, new_metrics=mse_metrics, metric_string_name=metric_string_name, epoch=epoch, where_to_log=where_to_log)

    metric_string_name = "kdmm"
    if metric_string_name in metrics_to_log:
        mse_0 = kmmd(generative_sample,test_sample)
        mse_metrics = {"mse_histograms_0": mse_0.item()}
        all_metrics = store_metrics(config, all_metrics, new_metrics=mse_metrics, metric_string_name=metric_string_name, epoch=epoch, where_to_log=where_to_log)

    """
    # METRICS
    metric_string_name = "graphs"
    if metric_string_name in metrics_to_log:

        graph_metrics = graph_metrics_for_oops(oops,generative_sample,test_sample)

        all_metrics = store_metrics(config, all_metrics, new_metrics=graph_metrics, metric_string_name=metric_string_name, epoch=epoch)

    # PLOTS
    metric_string_name = "graphs_plots"
    if metric_string_name in metrics_to_log:
        graph_plot_path_ = config.experiment_files.graph_plot_path.format(
            "_sinkhorn_{0}_{1}".format(sinkhorn_iteration, epoch))
        generated_graphs = oops.generate_graphs(number_of_graphs=20,
                                                generating_model=current_model,
                                                sinkhorn_iteration=1)
        plot_graphs_list2(generated_graphs, title="Generated 0", save_dir=graph_plot_path_)

    """
    return all_metrics