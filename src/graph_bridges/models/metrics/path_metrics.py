"""
These metrics that are applied for a paths when they appear in batches.

"""

import torch
from geomloss import SamplesLoss
from abc import ABC, abstractmethod
from torch.distributions import Bernoulli, Categorical, Multinomial


from graph_bridges.models.pipelines.sb.pipeline_sb import SBGBPipeline
from graph_bridges.models.metrics.data_metrics import SpinStatesStatistics
from graph_bridges.models.backward_rates.backward_rate import BackwardRate
from graph_bridges.models.metrics.data_metrics import SpinBernoulliMarginal

def wasserstein_between_paths_histograms(histogram_backward,histogram_forward):
    assert histogram_backward.shape == histogram_forward.shape
    loss = SamplesLoss("sinkhorn", p=2)
    number_of_steps = histogram_backward.shape[0]
    wasserstein_per_time = []
    for time_index in range(number_of_steps):
        dist = loss(histogram_backward[time_index, :].unsqueeze(1),
                    histogram_forward[time_index, :].unsqueeze(1))
        wasserstein_per_time.append(dist)
    return torch.Tensor(wasserstein_per_time)

def bernoulli_uniform_path(paths_dataloader,states=False):
    """
    generates a path where a uniform distribution is assigned to each spin

    :param paths_dataloader:
    :return:
    """
    paths_dataloader.number_of_time_steps
    sample_size = paths_dataloader.spin_dataloader_0.training_data_size
    number_of_spins = paths_dataloader.number_of_spins
    bernoulli_probability = torch.full((number_of_spins,), .5)

    bernoulli_sample = Bernoulli(bernoulli_probability).sample((sample_size,
                                                                paths_dataloader.number_of_time_steps + 1))
    spin_histogram = bernoulli_sample.sum(axis=0)
    bernoulli_paths_batch = (-1.)**bernoulli_sample
    if states:
        spin_statistics = SpinStatesStatistics(number_of_spins)
        states_histogram = spin_statistics.counts_states_in_paths(bernoulli_paths_batch)
        return bernoulli_paths_batch,states_histogram,spin_histogram
    else:
        return bernoulli_paths_batch, spin_histogram

def states_at_grid(paths_batch, times, time_grid):
    """
    For a simulation of states and times, for each value of the time grid,
    assigns the state with the rightmost time to the grid time.

    :param times: torch.Tensor(batch_size,number_of_timesteps)
    :param paths_batch torch.Tensor(batch_size,number_of_timesteps,number_of_spins):
    :param time_grid: torch.Tensor(number_of_timesteps)
    :return:
    """
    if len(times.shape) == 3:
        times = times.squeeze()

    assert torch.all(times[:, 0] == 0.)
    assert time_grid[0] == 0.
    assert paths_batch.shape[:-1] == times.shape

    batch_size = paths_batch.shape[0]
    tau = time_grid[1]
    time_grid_ = time_grid + tau # arrival is to the left of the box

    x = time_grid_[None, None, :] - times[:, :, None]
    times_not_allowed = times[:, :, None] > time_grid_[None, None, :]
    x[times_not_allowed] = torch.inf
    min_indices = torch.argmin(x, axis=1)

    arrivals = times[torch.arange(batch_size)[:, None], min_indices]
    states_at_arrival = paths_batch[torch.arange(batch_size)[:, None], min_indices, :]

    return states_at_arrival,arrivals

class PathMetric(ABC):
    """
    """
    name_ = "abstract_path_metric"
    def __init__(self,
                 paths_dataloader:SBGBPipeline=None,
                 current_model:BackwardRate=None,
                 past_model: BackwardRate=None,
                 forward:bool = False,
                 reference:bool = False,
                 parametric_form:str = "poisson_naive",
                 type="train",
                 **kwargs):
        self.paths_dataloader = paths_dataloader
        self.current_model = current_model
        self.past_model = past_model
        self.forward = forward
        self.reference = reference
        self.parametric_form = parametric_form
        self.type = type

    @abstractmethod
    def metric_on_pathbatch(self,path_batch,times,aggregation):
        return path_batch

    @abstractmethod
    def before_loop(self):
        return None

    @abstractmethod
    def after_loop(self):
        return None

    def after_loop_(self,aggregation):
        if self.type == "back_and_forward":
            past_aggregation = self.after_loop(aggregation[0])
            current_aggregation = self.after_loop(aggregation[1])
            return (past_aggregation,current_aggregation)
        else:
            return self.after_loop(aggregation)

    def before_loop_(self):
        if self.type == "back_and_forward":
            past_aggregation = self.before_loop()
            current_aggregation = self.before_loop()
            return (past_aggregation,current_aggregation)
        else:
            return self.before_loop()

    def metric_on_batch(self,batch,aggregation):
        if self.type != "back_and_forward":
            aggregation = self.metric_on_pathbatch(batch[0],batch[1],aggregation)
            return aggregation
        else:
            past_path = batch[0]
            past_time = batch[1]
            current_paths = batch[2]
            current_time = batch[3]
            past_aggregation = aggregation[0]
            current_aggregation = aggregation[1]
            past_aggregation = self.metric_on_pathbatch(past_path,past_time,past_aggregation)
            current_aggregation = self.metric_on_pathbatch(current_paths,current_time,current_aggregation)
            return (past_aggregation,current_aggregation)

    def __call__(self):
        aggregation = self.before_loop_()
        if self.type=="train":
            for batch in self.paths_dataloader.train(self.current_model,
                                                     self.forward,
                                                     self.reference,
                                                     self.parametric_form,
                                                     on_grid=True,
                                                     one_step=False):
                aggregation = self.metric_on_batch(batch,aggregation)
        elif self.type=="test":
            for batch in self.paths_dataloader.train(self.current_model,
                                                     self.forward,
                                                     self.reference,
                                                     self.parametric_form,
                                                     on_grid=True,
                                                     one_step=False):
                aggregation = self.metric_on_pathbatch(batch,aggregation)
        elif self.type=="back_and_forward":
            for batch in self.paths_dataloader.back_and_forward(self.current_model,
                                                                self.past_model,
                                                                self.forward,
                                                                self.reference):
                aggregation = self.metric_on_batch(batch,aggregation)
        else:
            raise Exception("Type of Data Not Included")

        final_results = self.after_loop_(aggregation)
        return final_results

class PathHistogram(PathMetric):
    name_="histogram_metric"
    def __init__(self,**kwargs):
        super(PathHistogram,self).__init__(**kwargs)
        self.spin_statistics = SpinStatesStatistics(self.paths_dataloader.spin_dataloader_0.number_of_spins)


    def metric_on_pathbatch(self,path_batch,times,histogram_of_states):
        if len(times.shape) == 3:
            path_batch, _ = states_at_grid(path_batch, times, self.paths_dataloader.time_grid)
        counts_per_states = self.spin_statistics.counts_states_in_paths(path_batch)
        histogram_of_states+=counts_per_states
        return histogram_of_states

    def before_loop(self):
        histogram_of_states = torch.zeros(self.paths_dataloader.number_of_time_steps + 1,
                                          self.spin_statistics.number_of_total_states)
        return histogram_of_states

    def after_loop(self,histogram_of_states):
        return histogram_of_states

class PathBernoulliMarginal(PathMetric):
    """
    """
    name_ = "bernoulli_marginal"
    def __init__(self, **kwargs):
        super(PathBernoulliMarginal, self).__init__(**kwargs)

    def metric_on_pathbatch(self,path_batch,times, aggregation):
        histogram_of_spins = aggregation[0]
        number_of_paths = aggregation[1]

        if len(times.shape) == 3:
            path_batch, _ = states_at_grid(path_batch, times, self.paths_dataloader.time_grid)

        path_batch[torch.where(path_batch == -1.)] = 0.
        number_of_paths += path_batch.shape[0]
        histogram_of_spins += path_batch.sum(axis=0)

        return (histogram_of_spins,number_of_paths)

    def before_loop(self):
        histogram_of_spins = torch.zeros(self.paths_dataloader.number_of_time_steps + 1,
                                         self.paths_dataloader.spin_dataloader_0.number_of_spins)
        return (histogram_of_spins,0.)

    def after_loop(self, aggregation):
        histogram_of_spins = aggregation[0]
        number_of_paths = aggregation[1]

        paths_counts = histogram_of_spins / number_of_paths
        assert torch.all(paths_counts <= 1.).item()

        #spins_sample = (-1.) ** (Bernoulli(paths_counts).sample(sample_shape=(number_of_paths,)) + 1.)

        return histogram_of_spins

def histogram_reference_path(paths_dataloader, paths_histogram):
    """
    generates a sample from the histogram

    :param paths_dataloader:
    :param paths_histogram:
    :return:
    """
    training_data_size = paths_dataloader.spin_dataloader_0.training_data_size
    sample_size = paths_histogram.sum(axis=1)[0].item()
    normalized_histogram = paths_histogram/sample_size
    assert torch.all(torch.round(normalized_histogram.sum(axis=1)) == 1.).item()
    categorical_from_histogram = Multinomial(training_data_size,normalized_histogram)
    counts_from_reference = categorical_from_histogram.sample()
    return counts_from_reference

def marginal_at_spins(paths_dataloader:SBGBPipeline,
                      current_model:BackwardRate,
                      past_model:BackwardRate,
                      **kwargs):
    """

    :param paths_dataloader:
    :param current_model:
    :param past_model:
    :param kwargs:

    :return:
    """
    sinkhorn_iteration = kwargs.get("sinkhorn_iteration")
    is_past_forward = kwargs.get("forward")
    reference = kwargs.get("reference")
    save_path_plot = kwargs.get("save_path_plot")
    plot = kwargs.get("plot",True)
    check_backward_learning = kwargs.get("check_backward_learning",False)

    number_of_paths = paths_dataloader.spin_dataloader_0.training_data_size

    # Bernoulli Histograms at the Ends
    bernoulli_histogram_at_0 = SpinBernoulliMarginal(spin_dataloader=paths_dataloader.spin_dataloader_0)()
    bernoulli_histogram_at_1 = SpinBernoulliMarginal(spin_dataloader=paths_dataloader.spin_dataloader_1)()
    spins_legends = list(map(str, range(paths_dataloader.number_of_spins)))

    #Bernoulli null model
    null_bernoulli_paths,  null_spin_histogram = bernoulli_uniform_path(paths_dataloader)

    past_current_bernoulli = PathBernoulliMarginal(paths_dataloader=paths_dataloader,
                                                   current_model=current_model,
                                                   past_model=past_model,
                                                   forward=is_past_forward,
                                                   reference=reference,
                                                   type="back_and_forward")
    past_current_bernoulli = past_current_bernoulli()

    histogram_from_rate = PathBernoulliMarginal(paths_dataloader=paths_dataloader,
                                                 current_model=current_model,
                                                 forward=not is_past_forward,
                                                 reference=False)
    histogram_from_rate = histogram_from_rate()
    time_ = paths_dataloader.time_grid

    #=========================================
    # WASSERSTEINS HISTOGRAMS
    #=========================================
    from graph_bridges.models.metrics.data_metrics import distances_at_ends

    wasserstein_at_ends_results = distances_at_ends(past_current_bernoulli,
                                                    histogram_from_rate,
                                                    bernoulli_histogram_at_0,
                                                    bernoulli_histogram_at_1,
                                                    null_spin_histogram[0,:],
                                                    is_past_forward,
                                                    number_of_paths)


    #w_null_to_forward = wasserstein_between_paths_histograms(path_histogram_0, path_histogram_1)
    #w_null_to_backward = wasserstein_between_paths_histograms(path_histogram_0, path_histogram_1)
    from graph_bridges.utils.plots.sb_plots import sinkhorn_plot

    if plot:
        sinkhorn_plot(sinkhorn_iteration,
                      is_past_forward,
                      time_,
                      bernoulli_histogram_at_0,
                      bernoulli_histogram_at_1,
                      past_current_bernoulli,
                      histogram_from_rate,
                      spins_legends,
                      check_backward_learning=check_backward_learning,
                      save_path=save_path_plot)

    return wasserstein_at_ends_results