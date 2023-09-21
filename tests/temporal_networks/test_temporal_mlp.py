import os
import sys
import torch
import unittest
from graph_bridges.configs.graphs.graph_config_ctdd import CTDDConfig
from graph_bridges.models.temporal_networks.mlp.temporal_mlp import TemporalMLP, TemporalMLPConfig
from graph_bridges.models.backward_rates.backward_rate_utils import load_backward_rates
from graph_bridges.models.backward_rates.ctdd_backward_rate import BackRateMLP

class TestConvnet(unittest.TestCase):

    def test_temporal_mlp(self):
        config = CTDDConfig()
        config.temp_network = TemporalMLPConfig()

        device = torch.device(config.trainer.device)

        model: BackRateMLP
        temp_net: TemporalMLP

        model = load_backward_rates(config,device)

        batch_size = 12
        D = config.data.D

        fake_image = torch.rand(batch_size,D).to(device)
        fake_time = torch.rand(batch_size).to(device)

        temp_net_forward = model.net(fake_image,fake_time)
        model_forward = model(fake_image,fake_time)

        print("temp_net")
        print(temp_net_forward.shape)
        print("backward_rate")
        print(model_forward.shape)


if __name__=="__main__":
    unittest.main()