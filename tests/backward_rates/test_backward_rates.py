import torch
import unittest
from dataclasses import asdict
from graph_bridges.models.generative_models.ctdd import CTDD

from graph_bridges.utils.tensor_padding import expand_with_zeros
from pprint import pprint

class TestUnetCifar10(unittest.TestCase):

    def test_unet_cifar10(self):
        from graph_bridges.configs.config_ctdd import CTDDTrainerConfig
        from graph_bridges.configs.images.cifar10_config_ctdd import CTDDConfig

        ctdd_config = CTDDConfig(experiment_indentifier="cifar10_wunet_test",
                                 experiment_name="cifar10",
                                 experiment_type="ctdd")
        ctdd_config.data.batch_size = 12
        device = torch.device(ctdd_config.trainer.device)
        ctdd = CTDD()
        ctdd.create_new_from_config(ctdd_config,device)

        databatch = next(ctdd.data_dataloader.train().__iter__())
        x_adj = databatch[0].to(device)
        batch_size = x_adj.shape[0]
        fake_time = torch.rand(batch_size).to(device)
        forward_net = ctdd.model.net(x_adj,fake_time)
        forward_ = ctdd.model(x_adj,fake_time)

        pprint(asdict(ctdd_config.model))
        print(f"Input shape: {x_adj.shape}")
        print(f"forward net shape: {forward_net.shape}")
        print(f"forward shape: {forward_.shape}")

class TestGraphs(unittest.TestCase):

    def test_backwardrate_mlp_graph(self):
        from graph_bridges.configs.graphs.graph_config_ctdd import CTDDConfig
        from graph_bridges.data.graph_dataloaders_config import CommunitySmallConfig

        from graph_bridges.models.temporal_networks.unets.unet_wrapper import UnetTauConfig
        from graph_bridges.models.temporal_networks.convnets.autoencoder import ConvNetAutoencoderConfig
        from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig
        from graph_bridges.models.backward_rates.ctdd_backward_rate_config import GaussianTargetRateImageX0PredEMAConfig

        ctdd_config = CTDDConfig(experiment_indentifier="graph_wunet_test",experiment_name="graph",experiment_type="ctdd")
        ctdd_config.data = CommunitySmallConfig(batch_size=24, full_adjacency=True)
        ctdd_config.model = BackRateMLPConfig()

        ctdd_config.data.batch_size = 12

        device = torch.device(ctdd_config.trainer.device)
        ctdd = CTDD()
        ctdd.create_new_from_config(ctdd_config,device)

        databatch = next(ctdd.data_dataloader.train().__iter__())
        x_adj = databatch[0].to(device)

        batch_size = x_adj.shape[0]
        fake_time = torch.rand(batch_size).to(device)
        print(f"Input shape: {x_adj.shape}")

        forward_ = ctdd.model(x_adj,fake_time)
        print(f"forward shape: {forward_.shape}")

        #forward_net = ctdd.model.net(x_adj, fake_time)
        #print(f"forward net shape: {forward_net.shape}")

    def test_convnet_graph(self):
        from graph_bridges.configs.graphs.graph_config_ctdd import CTDDConfig
        from graph_bridges.data.graph_dataloaders_config import CommunitySmallConfig
        from graph_bridges.models.backward_rates.ctdd_backward_rate_config import GaussianTargetRateImageX0PredEMAConfig
        from graph_bridges.models.temporal_networks.convnets.autoencoder import ConvNetAutoencoderConfig

        ctdd_config = CTDDConfig(experiment_indentifier="graph_wunet_test",
                                 experiment_name="graph",
                                 experiment_type="ctdd")
        ctdd_config.data = CommunitySmallConfig(batch_size=24, full_adjacency=True)
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
        print(f"Input shape: {x_adj.shape}")

        forward_ = ctdd.model(x_adj,fake_time)
        print(f"forward shape: {forward_.shape}")

        forward_net = ctdd.model.net(x_adj, fake_time)
        print(f"forward net shape: {forward_net.shape}")

    def test_temporal_hollow_transformers_graphs(self):
        from graph_bridges.configs.graphs.graph_config_ctdd import CTDDConfig
        from graph_bridges.data.graph_dataloaders_config import CommunitySmallConfig
        from graph_bridges.models.backward_rates.backward_rate_utils import load_backward_rates
        from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackwardRateTemporalHollowTransformerConfig
        from graph_bridges.models.temporal_networks.transformers.temporal_hollow_transformers import  TemporalHollowTransformerConfig

        config = CTDDConfig()
        config.data = CommunitySmallConfig(batch_size=24, full_adjacency=True)
        config.model = BackwardRateTemporalHollowTransformerConfig()
        config.temp_network = TemporalHollowTransformerConfig(num_layers=2,
                                                              num_heads=2,
                                                              hidden_dim=32,
                                                              ff_hidden_dim=64,
                                                              time_embed_dim=12,
                                                              time_scale_factor=10)


        device = torch.device(config.trainer.device)
        ctdd = CTDD()
        ctdd.create_new_from_config(config,device)

        databatch = next(ctdd.data_dataloader.train().__iter__())
        x_adj = databatch[0].to(device)

        batch_size = x_adj.shape[0]
        fake_time = torch.rand(batch_size).to(device)
        print(f"Input shape: {x_adj.shape}")

        forward_ = ctdd.model(x_adj,fake_time)
        print(f"forward shape: {forward_.shape}")

        print(forward_)

class TestNist(unittest.TestCase):

    def test_backwardrate_mlp_graph(self):

        from graph_bridges.configs.images.nist_config_ctdd import CTDDConfig
        from graph_bridges.models.temporal_networks.unets.unet_wrapper import UnetTauConfig
        from graph_bridges.data.graph_dataloaders_config import CommunitySmallConfig
        from graph_bridges.models.temporal_networks.convnets.autoencoder import ConvNetAutoencoderConfig
        from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig
        from graph_bridges.models.backward_rates.ctdd_backward_rate_config import GaussianTargetRateImageX0PredEMAConfig

        ctdd_config = CTDDConfig(experiment_indentifier="mnist_mlp_test",
                                 experiment_name="mnist",
                                 experiment_type="ctdd")
        ctdd_config.model = BackRateMLPConfig()
        ctdd_config.data.data = "mnist"

        device = torch.device(ctdd_config.trainer.device)

        ctdd = CTDD()
        ctdd.create_new_from_config(ctdd_config,device)

        databatch = next(ctdd.data_dataloader.train().__iter__())
        x_adj = databatch[0].to(device)

        batch_size = x_adj.shape[0]
        fake_time = torch.rand(batch_size).to(device)
        print(f"Input shape: {x_adj.shape}")

        forward_ = ctdd.model(x_adj,fake_time)
        print(f"forward shape: {forward_.shape}")

if __name__=="__main__":
    unittest.main()