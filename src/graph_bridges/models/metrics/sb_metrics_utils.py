import json
import torch
from graph_bridges.configs.config_sb import SBConfig
from graph_bridges.models.generative_models.sb import SB

from graph_bridges.models.metrics.sb_metrics import paths_marginal_histograms,sinkhorn_plot,graph_metrics_for_sb
from graph_bridges.models.metrics.sb_paths_metrics import states_paths_histograms_plots
from graph_bridges.models.metrics.histograms_metrics import marginals_histograms_mse
from graph_bridges.models.temporal_networks.graphs_networks.graph_plots import plot_graphs_list2
import torchvision

def log_metrics(sb:SB, current_model, past_to_train_model, sinkhorn_iteration, epoch, device, metrics_to_log=None,where_to_log=None,writer=None):
    """
    After the training procedure is done, the model is updated

    :return:
    """
    all_metrics = {}

    config = sb.config
    if metrics_to_log is None:
        metrics_to_log = config.trainer.metrics

    # HISTOGRAMS
    metric_string_name = "histograms"
    if metric_string_name in metrics_to_log:
        if where_to_log is None:
            histograms_plot_path_ = config.experiment_files.plot_path.format(
                metric_string_name + "_sinkhorn_{0}_{1}".format(sinkhorn_iteration,
                                                                epoch))
        else:
            histograms_plot_path_ = where_to_log[metric_string_name]

        all_histograms = paths_marginal_histograms(sb,
                                                   sinkhorn_iteration,
                                                   device,
                                                   current_model,
                                                   past_to_train_model,
                                                   config.trainer.exact_backward,
                                                   config.trainer.histograms_on_train)

        marginal_0, marginal_1, backward_histogram, forward_histogram, forward_time, state_legends = all_histograms

        sinkhorn_plot(sinkhorn_iteration,
                      marginal_0,
                      marginal_1,
                      backward_histogram=backward_histogram,
                      forward_histogram=forward_histogram,
                      time_=forward_time.cpu(),
                      states_legends=state_legends,
                      save_path=histograms_plot_path_)

        if writer is not None:
            # Read the saved image using torchvision
            image = torchvision.io.read_image(histograms_plot_path_)
            # Add the image to TensorBoard
            writer.add_image("matplotlib_plot", image, global_step=epoch)

        metric_string_name = "mse_histograms"
        if metric_string_name in metrics_to_log:
            marginal_histograms = marginal_0, backward_histogram[0, :], marginal_1, forward_histogram[-1, :]
            mse_1, mse_0 = marginals_histograms_mse(marginal_histograms)
            mse_metrics = {"mse_histograms_1": mse_1.tolist(),"mse_histograms_0": mse_0.tolist()}
            all_metrics.update(mse_metrics)
            mse_metric_path = config.experiment_files.metrics_file.format(metric_string_name + "_sinkhorn_{0}_{1}".format(sinkhorn_iteration, epoch))
            with open(mse_metric_path, "w") as f:
                json.dump(mse_metrics, f)

    else:
        metric_string_name = "mse_histograms"
        if metric_string_name in metrics_to_log:
            all_histograms = paths_marginal_histograms(sb,
                                                       sinkhorn_iteration,
                                                       device,
                                                       current_model,
                                                       past_to_train_model,
                                                       config.trainer.exact_backward,
                                                       config.trainer.histograms_on_train)

            marginal_0, marginal_1, backward_histogram, forward_histogram, forward_time, state_legends = all_histograms
            marginal_histograms = marginal_0, backward_histogram[0, :], marginal_1, forward_histogram[-1, :]
            mse_1, mse_0 = marginals_histograms_mse(marginal_histograms)
            mse_metrics = {"mse_histograms_1": mse_1.tolist(),"mse_histograms_0": mse_0.tolist()}
            mse_metric_path = config.experiment_files.metrics_file.format(metric_string_name + "_sinkhorn_{0}_{1}".format(sinkhorn_iteration,epoch))
            all_metrics.update(mse_metrics)
            with open(mse_metric_path, "w") as f:
                json.dump(mse_metrics, f)

    # METRICS
    metric_string_name = "graphs"
    if metric_string_name in metrics_to_log:
        graph_metrics_path_ = config.experiment_files.metrics_file.format("_sinkhorn_{0}_{1}".format(sinkhorn_iteration,
                                                                                                     epoch))
        graph_metrics = graph_metrics_for_sb(sb, current_model, device)
        all_metrics.update(graph_metrics)
        with open(graph_metrics_path_, "w") as f:
            json.dump(graph_metrics, f)

    # PLOTS
    metric_string_name = "graphs_plots"
    if metric_string_name in metrics_to_log:
        graph_plot_path_ = config.experiment_files.graph_plot_path.format(
            "_sinkhorn_{0}_{1}".format(sinkhorn_iteration, epoch))
        generated_graphs = sb.generate_graphs(number_of_graphs=20,
                                                   generating_model=current_model,
                                                   sinkhorn_iteration=1)
        plot_graphs_list2(generated_graphs, title="Generated 0", save_dir=graph_plot_path_)

    metric_string_name = "paths_histograms"
    if metric_string_name in metrics_to_log:
        if config.data.number_of_spins < 5:
            histograms_plot_path_2 = config.experiment_files.plot_path.format("_sinkhorn_{0}_{1}".format(sinkhorn_iteration, epoch))

            states_paths_histograms_plots(sb,
                                          sinkhorn_iteration=sinkhorn_iteration,
                                          device=device,
                                          current_model=current_model,
                                          past_to_train_model=past_to_train_model,
                                          plot_path=histograms_plot_path_2)

    return all_metrics