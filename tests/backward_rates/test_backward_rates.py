import torch
import unittest
import torch.nn as nn

from graph_bridges.configs.graphs.config_sb import SBConfig
from graph_bridges.data.graph_dataloaders_config import EgoConfig
from graph_bridges.configs.graphs.config_sb import SteinSpinEstimatorConfig,ParametrizedSamplerConfig

from graph_bridges.models.backward_rates.backward_rate_utils import  load_backward_rates
from graph_bridges.data.dataloaders_utils import load_dataloader


from graph_bridges.models.generative_models.sb import SB
from graph_bridges.data.graph_dataloaders import BridgeGraphDataLoaders
from graph_bridges.utils.test_utils import check_model_devices

class BaseBackwardRateForSBTest(object):
    """

    """

    data_loader: BridgeGraphDataLoaders
    sb_config: SBConfig
    sb: SB

    def backwardRateSetConfig(self):
        self.backward_rate_config = None  # To be overridden by subclasses

    def basicConfigSetUp(self):
        self.sb_config = SBConfig(experiment_indentifier="unittest",delete=True)
        self.backwardRateSetConfig()

        self.sb_config.model = self.backward_rate_config
        self.sb_config.data = EgoConfig(as_image=False,
                                        batch_size=5,
                                        full_adjacency=False,
                                        as_spins=False)
        self.sb_config.stein = SteinSpinEstimatorConfig(stein_sample_size=20)
        self.sb_config.sampler = ParametrizedSamplerConfig(num_steps=10)

    def basicSetUp(self):
        self.basicConfigSetUp()
        self.sb = SB(self.sb_config,torch.device("cpu"))

    def test_loading(self):
        print("Testing Loading")
        self.basicSetUp()
        self.assertIsNotNone(self.sb.training_model)

    def test_pipeline(self):
        print("Testing Pipeline")
        self.basicSetUp()
        x_end = self.sb.pipeline(self.sb.training_model, 1, torch.device("cpu"), return_path=False)
        self.assertIsInstance(x_end,torch.Tensor)

    def test_forward_pass(self):
        print("Testing Forward Pass")
        self.basicSetUp()
        databatch = next(self.sb.data_dataloader.train().__iter__())
        fake_time = self.sb.data_dataloader.fake_time_
        x_adj = databatch[0]

        forward_ = self.sb.training_model(x_adj,fake_time)
        forward_binary = self.sb.training_model.stein_binary_forward(x_adj, fake_time)

        self.assertIsInstance(forward_,torch.Tensor)
        self.assertIsInstance(forward_binary,torch.Tensor)

    def test_gpus(self):
        print("Testing GPUs")
        self.basicSetUp()
        device = torch.device("cuda:0")
        model = load_backward_rates(self.sb_config,device)
        model_device = check_model_devices(model)
        self.assertTrue(model_device == device)

        databatch = next(self.sb.data_dataloader.train().__iter__())
        x_adj = databatch[0]
        x_adj = x_adj.to(device)
        x_features =  databatch[1]
        times = self.sb.data_dataloader.fake_time_.to(device)

        forward_ = model(x_adj,times)
        forward_stein = model.stein_binary_forward(x_adj, times)

        self.assertTrue(forward_stein.device == device)
        self.assertTrue(forward_.device == device)
class TestBackRateMLP(BaseBackwardRateForSBTest,unittest.TestCase):

    def backwardRateSetConfig(self):
        from graph_bridges.models.backward_rates.backward_rate_config import BackRateMLPConfig
        self.backward_rate_config = BackRateMLPConfig()  # To be overridden by subclasses


class TestBackRateDoucetArchitecture(BaseBackwardRateForSBTest,unittest.TestCase):

    def backwardRateSetConfig(self):
        from graph_bridges.models.backward_rates.backward_rate_config import GaussianTargetRateImageX0PredEMAConfig
        self.backward_rate_config = GaussianTargetRateImageX0PredEMAConfig()  # To be overridden by subclasses


if __name__ == '__main__':
    unittest.main()

