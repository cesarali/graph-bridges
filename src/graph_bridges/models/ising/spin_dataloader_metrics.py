from discrete_diffusion.data.spins_dataloaders import SpinsDataLoader
from discrete_diffusion.models.ising.spin_states import SpinStatesStatistics

from abc import ABC, abstractmethod
import torch

class SpinDataloaderMetric(ABC):
    """
    """
    name_ = "abstract_spins_metric"
    def __init__(self,
                 spin_dataloader:SpinsDataLoader=None,
                 **kwargs):
        self.spin_dataloader = spin_dataloader

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
            for batch in self.paths_dataloader.test():
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

        batch[torch.where(batch == -1.)] = 0.
        number_of_paths += batch.shape[0]
        histogram_of_spins += batch.sum(axis=0)

        return (histogram_of_spins,number_of_paths)

    def before_loop(self):
        histogram_of_spins = torch.zeros(self.spin_dataloader.number_of_spins)
        return (histogram_of_spins,0.)

    def after_loop(self, aggregation):
        histogram_of_spins = aggregation[0]
        number_of_paths = aggregation[1]

        paths_counts = histogram_of_spins / number_of_paths
        assert torch.all(paths_counts <= 1.).item()

        #spins_sample = (-1.) ** (Bernoulli(paths_counts).sample(sample_shape=(number_of_paths,)) + 1.)

        return histogram_of_spins

if __name__=="__main__":
    from discrete_diffusion.data.spins_dataloaders import BernoulliSpinsDataLoader
    from matplotlib import pyplot as plt

    Q_sigma = 512.
    batch_size = 32
    number_of_spins = 3
    number_of_paths = 500

    # BERNOULLI
    bernoulli_dataloader_parameters_0 = {'batch_size': batch_size,
                                       'bernoulli_probability': 0.1,
                                       'number_of_paths': number_of_paths,
                                       'number_of_spins': number_of_spins,
                                       'remove': False}
    bernoulli_dataloader_0 = BernoulliSpinsDataLoader(**bernoulli_dataloader_parameters_0)

    bernoulli_dataloader_parameters_1 = {'batch_size': batch_size,
                                       'bernoulli_probability': 0.9,
                                       'number_of_paths': number_of_paths,
                                       'number_of_spins': number_of_spins,
                                       'remove': False}
    bernoulli_dataloader_1 = BernoulliSpinsDataLoader(**bernoulli_dataloader_parameters_1)

    #histogram_metric_0 = SpinStateHistogram(spin_dataloader=bernoulli_dataloader_0)
    bernoulli_marginal = SpinBernoulliMarginal(spin_dataloader=bernoulli_dataloader_0)

    #histogram_0 = histogram_metric_0()
    marginal_0 = bernoulli_marginal()

    #plt.bar([a for a in range(len(histogram_0.tolist()))],histogram_0.tolist())
    plt.bar([a for a in range(bernoulli_dataloader_0.number_of_spins)],marginal_0.tolist())
    plt.show()
