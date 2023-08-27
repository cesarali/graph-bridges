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
        self.data_config = EgoConfig(as_image=False,batch_size=5,full_adjacency=False,as_spins=False)

    def basicConfigSetUp(self):
        self.sb_config = SBConfig(experiment_indentifier="backward_rates_unittest",delete=True)
        self.backwardRateSetConfig()

        self.sb_config.model = self.backward_rate_config
        self.sb_config.data = self.data_config
        self.sb_config.stein = SteinSpinEstimatorConfig(stein_sample_size=2)
        self.sb_config.sampler = ParametrizedSamplerConfig(num_steps=10)
        self.device = torch.device("cpu")

    def basicSetUp(self):
        self.basicConfigSetUp()
        self.sb = SB()
        self.sb.create_new_from_config(self.sb_config,self.device)

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
        self.assertFalse(torch.isnan(forward_).any())
        self.assertFalse(torch.isinf(forward_).any())

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
        self.data_config = EgoConfig(as_image=False,batch_size=2,full_adjacency=False,as_spins=False)
        self.backward_rate_config = BackRateMLPConfig()  # To be overridden by subclasses

class TestBackRateDoucetArchitecture(BaseBackwardRateForSBTest,unittest.TestCase):

    def backwardRateSetConfig(self):
        from graph_bridges.models.backward_rates.backward_rate_config import GaussianTargetRateImageX0PredEMAConfig
        from graph_bridges.data.graph_dataloaders_config import PepperMNISTDataConfig

        self.data_config = EgoConfig(as_image=False,batch_size=5,full_adjacency=False)
        #self.data_config =  PepperMNISTDataConfig(as_image=False, batch_size=2, full_adjacency=False)
        self.backward_rate_config = GaussianTargetRateImageX0PredEMAConfig()  # To be overridden by subclasses



class TestBackwardRateForCifar10(unittest.TestCase):

    def test_cifar10(self):
        from graph_bridges.models.backward_rates.backward_rate_config import GaussianTargetRateImageX0PredEMAConfig
        from graph_bridges.data.image_dataloader_config import DiscreteCIFAR10Config
        from graph_bridges.configs.images.cifar10_config_ctdd import CTDDConfig
        from graph_bridges.models.generative_models.ctdd import CTDD
        from graph_bridges.models.backward_rates.backward_rate_utils import load_backward_rates
        config = CTDDConfig()
        config.data = DiscreteCIFAR10Config()
        config.trainer.device = "cpu"

        #device
        device = torch.device(config.trainer.device)

        #dataloader
        #dataloader = load_dataloader(config,device=torch.device("cpu"))
        #databath = next(dataloader.train().__iter__())
        #x = databath[0]

        x = torch.randint(255,(128,3,32,32))

        #model
        #ctdd = CTDD()
        #ctdd.create_new_from_config(config, device)
        config.target.S = config.data.S
        config.target.D = config.data.D

        model = load_backward_rates(config,device)
        fake_time = torch.rand(x.shape[0])

        forward_pass = model(x,fake_time)
        self.assertIsNotNone(forward_pass)


if __name__ == '__main__':
    unittest.main()

