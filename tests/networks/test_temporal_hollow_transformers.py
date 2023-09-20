import torch
import unittest


class TestTemporalHollowTransformers(unittest.TestCase):

    def test_temporal_hollow_transformers(self):
        from graph_bridges.configs.graphs.graph_config_ctdd import CTDDConfig
        from graph_bridges.data.graph_dataloaders_config import CommunitySmallConfig
        from graph_bridges.models.backward_rates.backward_rate_utils import load_backward_rates
        from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackwardRateTemporalHollowTransformerConfig

        from graph_bridges.models.temporal_networks.transformers.temporal_hollow_transformers import TemporalHollowTransformerConfig

        config = CTDDConfig()
        device = torch.device(config.trainer.device)
        config.data = CommunitySmallConfig(batch_size=24, full_adjacency=True)
        config.model = BackwardRateTemporalHollowTransformerConfig()


        model: BackwardRateTemporalHollowTransformerConfig
        input_vocab_size = output_vocab_size = 14
        max_seq_length = D = 10
        batch_size = 32

        config.temp_network = TemporalHollowTransformerConfig(input_vocab_size=input_vocab_size,
                                                              output_vocab_size=output_vocab_size,
                                                              max_seq_length=max_seq_length,
                                                              num_layers=2,
                                                              num_heads=2,
                                                              hidden_dim=12,
                                                              ff_hidden_dim=24,
                                                              time_embed_dim=12,
                                                              time_scale_factor=10)


        model: BackwardRateTemporalHollowTransformerConfig
        model = load_backward_rates(config,device)

        x_adj = torch.randint(0, input_vocab_size, (batch_size, max_seq_length)).to(device)
        time = torch.rand(batch_size).to(device)
        forward_model = model(x_adj,time)

        print(f"x_adj shape: {x_adj.shape}")
        print(f"time shape: {time.shape}")
        print(f"forward_model shape: {forward_model.shape}")

if __name__=="__main__":
    unittest.main()