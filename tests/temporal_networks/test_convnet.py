import os
import sys
import torch
import unittest
from graph_bridges.configs.images.nist_config_ctdd import CTDDConfig
from graph_bridges.models.temporal_networks.convnets.autoencoder import ConvNetAutoencoderConfig
from graph_bridges.models.backward_rates.backward_rate_utils import load_backward_rates
from graph_bridges.models.backward_rates.ctdd_backward_rate import GaussianTargetRateImageX0PredEMA

class TestConvnet(unittest.TestCase):

    def test_convnet(self):
        config = CTDDConfig()
        config.data.data = "mnist"
        device = torch.device(config.trainer.device)

        model: GaussianTargetRateImageX0PredEMA
        model = load_backward_rates(config,device)

        batch_size = 12
        in_channels = config.data.C
        height = config.data.H
        width =config.data.W

        fake_image = torch.rand(batch_size,in_channels,height,width).to(device)
        fake_time = torch.rand(batch_size).to(device)

        temp_net_forward = model.net(fake_image,fake_time)
        model_forward = model(fake_image,fake_time)

        print("temp_net")
        print(temp_net_forward.shape)
        print("backward_rate")
        print(model_forward.shape)


if __name__=="__main__":
    unittest.main()