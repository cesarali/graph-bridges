"""
These are metrics that are applied to the full data contained in a dataloader
the aim is to aggregate metrics when we only have access to different batches

"""
from torch.distributions import Bernoulli
from graph_bridges.data.graph_dataloaders import BridgeGraphDataLoaders
from graph_bridges.models.spin_glass.spin_states import SpinStatesStatistics
from abc import ABC, abstractmethod
import torch
from torch.distributions.kl import kl_divergence
from geomloss import SamplesLoss

def kl_for_bernoulli(p1,p2,reduce="sum"):
    p1[p1 == 1.] = 0.9999
    p2[p2 == 1.] = 0.9999

    d1 = Bernoulli(p1)
    d2 = Bernoulli(p2)
    kl_ = kl_divergence(d1, d2)
    if reduce == "sum":
        return kl_.sum()
    elif reduce == "mean":
        return kl_.mean()


def distances_at_ends(past_current_states,
                      histogram_from_rate,
                      bernoulli_histogram_at_0,
                      bernoulli_histogram_at_1,
                      null_histograms_bernoulli,
                      forward,
                      number_of_paths):
    """

    :param past_current_states:
    :param histogram_from_rate:
    :param bernoulli_histogram_at_0:
    :param bernoulli_histogram_at_1:
    :param histograms_bernoulli:
    :param forward:
    :return:
    """
    loss = SamplesLoss("sinkhorn", p=2)
    start_real = bernoulli_histogram_at_0
    end_real = bernoulli_histogram_at_1

    if forward:
        forward_histogram = past_current_states[0]
        backward_histogram = histogram_from_rate

        backward_histogram_check = past_current_states[1]
        start_target_check = backward_histogram_check[0,:]

        wasserstein_forward_backward = loss(start_real.unsqueeze(1),
                                     start_target_check.unsqueeze(1))

        kl_forward_backward = kl_for_bernoulli(start_real/number_of_paths,start_target_check/number_of_paths)
    else:
        backward_histogram = past_current_states[0]
        forward_histogram = histogram_from_rate

        forward_histogram_check = past_current_states[1]
        end_target_check = forward_histogram_check[-1, :]

        wasserstein_forward_backward = loss(end_real.unsqueeze(1),end_target_check.unsqueeze(1))
        kl_forward_backward = kl_for_bernoulli(end_real/number_of_paths,end_target_check/number_of_paths)

    start_target = backward_histogram[0,:]
    end_target = forward_histogram[-1,:]

    wasserstein_end_null = loss(end_real.unsqueeze(1),
                                null_histograms_bernoulli.unsqueeze(1))

    wasserstein_start_null = loss(start_real.unsqueeze(1),
                                  null_histograms_bernoulli.unsqueeze(1))

    wasserstein_start = loss(start_target.unsqueeze(1),
                      start_real.unsqueeze(1))

    wasserstein_end = loss(end_target.unsqueeze(1),
                           end_real.unsqueeze(1))

    kl_end_null = kl_for_bernoulli(end_real/number_of_paths,
                                null_histograms_bernoulli/number_of_paths)

    kl_start_null = kl_for_bernoulli(start_real/number_of_paths,
                                  null_histograms_bernoulli/number_of_paths)

    kl_start = kl_for_bernoulli(start_target/number_of_paths,
                                start_real/number_of_paths)

    kl_end = kl_for_bernoulli(end_target/number_of_paths,
                              end_real/number_of_paths)

    kls = {"kl_end_null":kl_end_null,
           "kl_start_null":kl_start_null,
           "kl_start":kl_start,
           "kl_end":kl_end,
           "kl_start_normalized":kl_start/kl_start_null,
           "kl_end_normalized":kl_end/kl_end_null,
           "kl_forward_backward":kl_forward_backward,
           "kl_forward_backward_normalized":(2.*kl_forward_backward)/(kl_start_null + kl_end_null)}

    wassersteins = {"wasserstein_start": wasserstein_start,
                   "wasserstein_end": wasserstein_end,
                   "wasserstein_forward_backward": wasserstein_forward_backward,
                   "wasserstein_start_real_to_null": wasserstein_start_null,
                   "wasserstein_end_real_to_null": wasserstein_end_null,
                   "wasserstein_start_normalized": wasserstein_start / wasserstein_start_null,
                   "wasserstein_end_normalized": wasserstein_end / wasserstein_end_null,
                   "wasserstein_forward_backward_normalized": (2. * wasserstein_forward_backward) / (wasserstein_start_null + wasserstein_end_null)}

    results = {"wasserstein":wassersteins,
               "kl_distances":kls}

    return results


class SpinDataloaderMetric(ABC):
    """
    """
    name_ = "abstract_spins_metric"
    def __init__(self,
                 spin_dataloader:BridgeGraphDataLoaders=None,
                 device=torch.device("cpu"),
                 **kwargs):
        self.spin_dataloader = spin_dataloader
        self.doucet = spin_dataloader.doucet
        self.device = device

    @abstractmethod
    def metric_on_pathbatch(self,batch,aggregation):
        return batch

    @abstractmethod
    def before_loop(self):
        return None

    @abstractmethod
    def after_loop(self):
        return None

    def __call__(self,type="train"):
        aggregation = self.before_loop()
        if type=="train":
            for batch in self.spin_dataloader.train():
                aggregation = self.metric_on_pathbatch(batch,aggregation)
        elif type=="test":
            for batch in self.spin_dataloader.test():
                aggregation = self.metric_on_pathbatch(batch,aggregation)
        else:
            raise Exception("Type of Data Not Included")
        final_results = self.after_loop(aggregation)
        return final_results

class SpinStateHistogram(SpinDataloaderMetric):
    name_="histogram_metric"
    def __init__(self,**kwargs):
        super(SpinStateHistogram,self).__init__(**kwargs)
        self.spin_statistics = SpinStatesStatistics(self.spin_dataloader.number_of_spins)

    def metric_on_pathbatch(self,batch,histogram_of_states):
        counts_per_states = self.spin_statistics.counts_for_different_states(batch[0])
        histogram_of_states+=counts_per_states
        return histogram_of_states

    def before_loop(self):
        histogram_of_states = torch.zeros(self.spin_statistics.number_of_total_states)
        return histogram_of_states

    def after_loop(self,histogram_of_states):
        return histogram_of_states

class SpinBernoulliMarginal(SpinDataloaderMetric):
    """
    """
    name_ = "bernoulli_marginal"
    def __init__(self, **kwargs):
        super(SpinBernoulliMarginal, self).__init__(**kwargs)

    def metric_on_pathbatch(self, batch, aggregation):
        batch = batch[0]
        histogram_of_spins = aggregation[0]
        number_of_paths = aggregation[1]
        if not self.doucet:
            batch[torch.where(batch == -1.)] = 0.
        else:
            batch = batch.squeeze()

        number_of_paths += batch.shape[0]
        if len(batch.shape) >= 2:
            batch = batch.reshape(batch.shape[0],-1)
        histogram_of_spins += batch.to("cpu").sum(axis=0)

        return (histogram_of_spins,number_of_paths)

    def before_loop(self):
        histogram_of_spins = torch.zeros(self.spin_dataloader.number_of_spins,device=self.device)
        return (histogram_of_spins,0.)

    def after_loop(self, aggregation):
        histogram_of_spins = aggregation[0]
        number_of_paths = aggregation[1]

        paths_counts = histogram_of_spins / number_of_paths
        print(f"from marginal {number_of_paths}")
        assert torch.all(paths_counts <= 1.).item()

        #spins_sample = (-1.) ** (Bernoulli(paths_counts).sample(sample_shape=(number_of_paths,)) + 1.)

        return histogram_of_spins


if __name__=="__main__":
    from graph_bridges.configs.graphs.graph_config_ctdd import CTDDConfig
    from graph_bridges.data.graph_dataloaders_config import EgoConfig

    from graph_bridges.data.dataloaders_utils import load_dataloader

    device = torch.device("cpu")

    ctdd_config = CTDDConfig()
    ctdd_config.data = EgoConfig(as_image=False)

    data_loader = load_dataloader(ctdd_config,type="data",device=device)
    marginal_0 = SpinBernoulliMarginal(spin_dataloader=data_loader)()

