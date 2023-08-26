import torch
import unittest
from graph_bridges.data.image_dataloader_config import DiscreteCIFAR10Config
from graph_bridges.configs.images.config_ctdd import CTDDConfig
from graph_bridges.data.dataloaders_utils import load_dataloader
class TestCIFAR10(unittest.TestCase):

    def test_cifar10(self):
        config = CTDDConfig
        config.data = DiscreteCIFAR10Config()
        dataloader = load_dataloader(config,device=torch.device("cpu"))
        databath = next(dataloader.train().__iter__())

if __name__=="__main__":
    unittest.main()