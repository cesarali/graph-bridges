import torch
import unittest

from graph_bridges.data.dataloaders_utils import load_dataloader
from graph_bridges.configs.graphs.graph_config_ctdd import CTDDConfig
from graph_bridges.data.graph_dataloaders_config import EgoConfig, GraphSpinsDataLoaderConfig, TargetConfig
from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig, GaussianTargetRateImageX0PredEMAConfig
from graph_bridges.models.reference_process.ctdd_reference import GaussianTargetRate

class TestGaussianReference(unittest.TestCase):

    def setUp(self) -> None:
        config = CTDDConfig(experiment_indentifier="test_1")
        config.data = EgoConfig(as_image=False, batch_size=5, full_adjacency=False)
        config.model = GaussianTargetRateImageX0PredEMAConfig(time_embed_dim=32, fix_logistic=False)
        config.initialize_new_experiment()

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.data_dataloader = load_dataloader(config, type="data", device=self.device)
        self.target_dataloader = load_dataloader(config, type="target", device=self.device)

        self.x_adj_data = next(self.data_dataloader.train().__iter__())[0].to(self.device)
        self.x_adj_target = next(self.target_dataloader.train().__iter__())[0].to(self.device)

        self.batch_size = self.x_adj_data.shape[0]
        self.times = torch.rand(self.batch_size,device=self.device)

        self.reference_process = GaussianTargetRate(config, self.device)

    def test_rates_and_probabilities(self):
        stein_binary_forward = self.reference_process.stein_binary_forward(states=self.x_adj_target, times=self.times)
        forward_ = self.reference_process.forward_rates(self.x_adj_data, self.times, self.device)
        rate_ = self.reference_process.rate(self.times)
        transition_ = self.reference_process.transition(self.times)
        forward_rates_and_probabilities_ = self.reference_process.forward_rates_and_probabilities(self.x_adj_data, self.times, self.device)
        forward_rates, qt0_denom, qt0_numer = forward_rates_and_probabilities_
        forward_rates_and_probabilities_ = self.reference_process.forward_rates_and_probabilities(self.x_adj_target, self.times, self.device)
        forward_rates, qt0_denom, qt0_numer = forward_rates_and_probabilities_
        forward_rates_ = self.reference_process.forward_rates(self.x_adj_data, self.times, self.device)
        forward_rates_ = self.reference_process.forward_rates(self.x_adj_target, self.times, self.device)
        print(
            "rate_ {0}, transition_ {1}, forward_rates {2}, qt0_denom {3}, qt0_numer {4}, forward_rates_ {5}, ".format(
                rate_.shape, transition_.shape, forward_rates.shape, qt0_denom.shape, qt0_numer.shape,
                forward_rates_.shape))

    def test_rates_and_probabilities_device(self):
        stein_binary_forward = self.reference_process.stein_binary_forward(states=self.x_adj_data, times=self.times)
        forward_ = self.reference_process.forward_rates(self.x_adj_data, self.times, self.device)
        rate_ = self.reference_process.rate(self.times)
        transition_ = self.reference_process.transition(self.times)
        forward_rates_and_probabilities_ = self.reference_process.forward_rates_and_probabilities(self.x_adj_data, self.times, self.device)
        forward_rates, qt0_denom, qt0_numer = forward_rates_and_probabilities_
        forward_rates_ = self.reference_process.forward_rates(self.x_adj_data, self.times, self.device)

        self.assertTrue(self.device == stein_binary_forward.device)
        self.assertTrue(self.device == forward_.device)
        self.assertTrue(self.device == rate_.device)
        self.assertTrue(self.device == transition_.device)
        self.assertTrue(self.device == forward_rates.device)
        self.assertTrue(self.device == qt0_denom.device)
        self.assertTrue(self.device == qt0_numer.device)
        self.assertTrue(self.device == forward_rates_.device)

if __name__=="__main__":
    unittest.main()
