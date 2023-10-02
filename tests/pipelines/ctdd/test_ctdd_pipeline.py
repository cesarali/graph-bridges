import os
import torch
import unittest
from graph_bridges.models.generative_models.ctdd import CTDD

from dataclasses import asdict
from pprint import pprint

class TestUnetCifar10(unittest.TestCase):

    def test_unet_cifar10(self):
        from graph_bridges.configs.config_ctdd import CTDDTrainerConfig
        from graph_bridges.configs.images.cifar10_config_ctdd import CTDDConfig as Cifar10CTDDConfig

        ctdd_config = Cifar10CTDDConfig(experiment_indentifier="cifar10_wunet_test",
                                        experiment_name="cifar10",
                                        experiment_type="ctdd")
        ctdd_config.data.batch_size = 12
        ctdd_config.sampler.num_steps=2

        device = torch.device(ctdd_config.device)
        ctdd = CTDD()
        ctdd.create_new_from_config(ctdd_config,device)

        databatch = next(ctdd.data_dataloader.train().__iter__())
        x_adj:torch.Tensor = databatch[0]
        batch_size = x_adj.shape[0]

        fake_time = torch.rand(batch_size)
        forward_ = ctdd.model(x_adj,fake_time)
        forward_net = ctdd.model.net(x_adj,fake_time)

        print(asdict(ctdd_config.model))
        print(f"Data Sample Shape {x_adj.shape}")
        print(f"Forward Backward Rate Model {forward_.shape}")
        print(f"Forward Net Model {forward_net.shape}")

        x_sample = ctdd.pipeline(ctdd.model,sample_size=32,device=device)
        print(f"Pipeline sample shape {x_sample.shape}")

class TestUnetGraph(unittest.TestCase):

    def test_unet_graph(self):
        from graph_bridges.configs.graphs.graph_config_ctdd import CTDDConfig
        from graph_bridges.data.graph_dataloaders_config import CommunitySmallConfig
        from graph_bridges.models.temporal_networks.convnets.autoencoder import ConvNetAutoencoderConfig
        from graph_bridges.models.backward_rates.ctdd_backward_rate_config import GaussianTargetRateImageX0PredEMAConfig

        ctdd_config = CTDDConfig(experiment_indentifier="graph_wunet_test",experiment_name="graph",experiment_type="ctdd")
        ctdd_config.data = CommunitySmallConfig(batch_size=24)
        ctdd_config.model = GaussianTargetRateImageX0PredEMAConfig()
        ctdd_config.temp_network = ConvNetAutoencoderConfig()

        ctdd_config.data.batch_size = 12
        device = torch.device(ctdd_config.trainer.device)

        ctdd = CTDD()
        ctdd.create_new_from_config(ctdd_config,device)

        databatch = next(ctdd.data_dataloader.train().__iter__())
        x_adj = databatch[0].to(device)
        batch_size = x_adj.shape[0]
        fake_time = torch.rand(batch_size).to(device)
        print(asdict(ctdd_config.model))
        print(f"Data Sample Shape {x_adj.shape}")
        forward_ = ctdd.model(x_adj,fake_time)
        forward_net = ctdd.model.net(x_adj,fake_time)
        x_sample = ctdd.pipeline(ctdd.model,sample_size=batch_size,device=device)

        print(f"Forward Backward Rate Model {forward_.shape}")
        print(f"Forward Net Model {forward_net.shape}")
        print(f"Pipeline sample shape {x_sample.shape}")

class TestConvNetNist(unittest.TestCase):

    def test_convnet_mnist(self):
        from graph_bridges.configs.images.nist_config_ctdd import CTDDConfig
        from graph_bridges.data.image_dataloader_config import NISTLoaderConfig
        from graph_bridges.models.temporal_networks.convnets.autoencoder import ConvNetAutoencoderConfig
        from graph_bridges.models.backward_rates.ctdd_backward_rate_config import GaussianTargetRateImageX0PredEMAConfig

        ctdd_config = CTDDConfig(experiment_indentifier="graph_convnet_test",
                                 experiment_name="mnist",
                                 experiment_type="ctdd")
        ctdd_config.data = NISTLoaderConfig(batch_size=24)

        ctdd_config.model = GaussianTargetRateImageX0PredEMAConfig()
        ctdd_config.temp_network = ConvNetAutoencoderConfig()

        ctdd_config.data.batch_size = 12
        device = torch.device(ctdd_config.trainer.device)

        ctdd = CTDD()
        ctdd.create_new_from_config(ctdd_config,device)

        databatch = next(ctdd.data_dataloader.train().__iter__())
        x_adj = databatch[0].to(device)
        batch_size = x_adj.shape[0]
        fake_time = torch.rand(batch_size).to(device)
        pprint(asdict(ctdd_config.temp_network))

        print(f"Data Sample Shape {x_adj.shape}")
        forward_ = ctdd.model(x_adj,fake_time)
        forward_net = ctdd.model.net(x_adj,fake_time)
        x_sample = ctdd.pipeline(ctdd.model,sample_size=batch_size,device=device)

        print(f"Forward Backward Rate Model {forward_.shape}")
        print(f"Forward Net Model {forward_net.shape}")
        print(f"Pipeline sample shape {x_sample.shape}")

class TestMLPNist(unittest.TestCase):

    def test_mlp_mnist(self):
        from graph_bridges.configs.images.nist_config_ctdd import CTDDConfig
        from graph_bridges.data.image_dataloader_config import NISTLoaderConfig

        from graph_bridges.models.temporal_networks.convnets.autoencoder import ConvNetAutoencoderConfig
        from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig

        ctdd_config = CTDDConfig(experiment_indentifier="graph_mlp_test",
                                 experiment_name="mnist",
                                 experiment_type="ctdd")
        ctdd_config.data = NISTLoaderConfig(batch_size=24)
        ctdd_config.model = BackRateMLPConfig()

        ctdd_config.data.batch_size = 12
        device = torch.device(ctdd_config.trainer.device)

        ctdd = CTDD()
        ctdd.create_new_from_config(ctdd_config, device)

        databatch = next(ctdd.data_dataloader.train().__iter__())
        x_adj = databatch[0].to(device)
        batch_size = x_adj.shape[0]
        fake_time = torch.rand(batch_size).to(device)
        pprint(asdict(ctdd_config.temp_network))

        print(f"Data Sample Shape {x_adj.shape}")
        forward_ = ctdd.model(x_adj, fake_time)
        x_sample = ctdd.pipeline(ctdd.model, sample_size=batch_size, device=device)

        print(f"Forward Backward Rate Model {forward_.shape}")
        print(f"Pipeline sample shape {x_sample.shape}")

if __name__=="__main__":
    unittest.main()