from graph_bridges.data.image_dataloader_config import DiscreteCIFAR10Config
from graph_bridges.configs.images.cifar10_config_ctdd import CTDDConfig
from graph_bridges.data.dataloaders_utils import load_dataloader
import unittest
import torch
from dataclasses import asdict
from pprint import pprint

class TestParametrizedHamiltonian(unittest.TestCase):

    def test_spin_glass(self):
        from graph_bridges.configs.spin_glass.spin_glass_config_ctdd import CTDDConfig
        config = CTDDConfig()
        config.data.bernoulli_spins = True
        config.data.bernoulli_probability = 0.9
        config.data.number_of_paths = 1200
        config.data.data = f"bernoulli_probability_{config.data.bernoulli_probability}"
        config.data.__post_init__()
        pprint(asdict(config.data))
        config.data.batch_size = 32
        dataloader = load_dataloader(config,device=torch.device("cpu"))
        databath = next(dataloader.train().__iter__())
        print(databath[0].shape)
        print(len(databath[0]))

    def test_spins_size(self):
        from graph_bridges.configs.spin_glass.spin_glass_config_ctdd import CTDDConfig
        from graph_bridges.data.spin_glass_dataloaders_config import ParametrizedSpinGlassHamiltonianConfig

        config = CTDDConfig()
        config.data = ParametrizedSpinGlassHamiltonianConfig(data="bernoulli_probability_0.2")
        dataloader = load_dataloader(config,device=torch.device("cpu"))
        #databatch = 0
        #for databatch in dataloader.train():
        print(config.data.total_data_size)

        config = CTDDConfig()
        config.data = ParametrizedSpinGlassHamiltonianConfig(data="bernoulli_probability_0.9")
        dataloader = load_dataloader(config, device=torch.device("cpu"))
        print(config.data.total_data_size)




if __name__=="__main__":
    unittest.main()