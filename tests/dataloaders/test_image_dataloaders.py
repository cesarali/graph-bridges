from graph_bridges.data.image_dataloader_config import DiscreteCIFAR10Config
from graph_bridges.configs.images.cifar10_config_ctdd import CTDDConfig
from graph_bridges.data.dataloaders_utils import load_dataloader

import unittest
import torch

class TestCIFAR10(unittest.TestCase):

    def test_cifar10(self):
        from graph_bridges.configs.images.cifar10_config_ctdd import CTDDConfig
        config = CTDDConfig
        config.data = DiscreteCIFAR10Config()
        dataloader = load_dataloader(config,device=torch.device("cpu"))
        databath = next(dataloader.train().__iter__())
        print(databath[0].shape)

class TestNIST(unittest.TestCase):

    def test_mnist(self):
        from graph_bridges.configs.images.nist_config_ctdd import CTDDConfig
        config = CTDDConfig()
        config.data.data = "fashion"
        dataloader = load_dataloader(config,device=torch.device("cpu"))
        databath = next(dataloader.train().__iter__())
        x_adj = databath[0]
        data_config = config.data
        print(data_config.total_data_size)
        print(x_adj.shape)

if __name__=="__main__":
    unittest.main()