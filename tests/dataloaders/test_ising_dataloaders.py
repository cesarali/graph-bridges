from graph_bridges.data.image_dataloader_config import DiscreteCIFAR10Config
from graph_bridges.configs.images.cifar10_config_ctdd import CTDDConfig
from graph_bridges.data.dataloaders_utils import load_dataloader
import unittest
import torch

class TestParametrizedHamiltonian(unittest.TestCase):

    def test_spin_glass(self):
        from graph_bridges.configs.spin_glass.spin_glass_config_ctdd import CTDDConfig
        config = CTDDConfig()
        config.data.batch_size = 32
        dataloader = load_dataloader(config,device=torch.device("cpu"))
        databath = next(dataloader.train().__iter__())
        print(databath[0].shape)
        print(len(databath[0]))


if __name__=="__main__":
    unittest.main()