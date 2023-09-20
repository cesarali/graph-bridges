import torch
import unittest

from graph_bridges.models.temporal_networks.transformers.hollow_transformers import MultiheadHollowAttention
from graph_bridges.models.temporal_networks.transformers.hollow_transformers import HollowTransformer

class TestHollowTransformers(unittest.TestCase):

    def test_multihead_hollow_attention(self):
        # Example usage
        d_model = 256
        num_heads = 8
        seq_length = 10
        batch_size = 4

        # Creating random input tensors
        query = torch.randn(batch_size, seq_length, d_model)
        key = torch.randn(batch_size, seq_length, d_model)
        value = torch.randn(batch_size, seq_length, d_model)

        # Creating a random hollow mask
        mask = torch.zeros(batch_size, seq_length, seq_length)
        mask[0, :, :5] = 1
        mask[1, :, :8] = 1
        mask[2, :, :3] = 1
        mask[3, :, :6] = 1

        attention = MultiheadHollowAttention(num_heads, d_model)
        output = attention(query, key, value, mask)
        self.assertIsNotNone(output)

    def test_hollow_tranformer(self):
        # Example usage
        num_layers = 4
        num_heads = 8
        d_model = 256
        ff_hidden_dim = 512
        input_vocab_size = 2
        output_vocab_size = 2
        max_seq_length = 50
        batch_size = 16

        # Creating random input tensor
        input_data = torch.randint(0, input_vocab_size, (batch_size, max_seq_length))

        # Creating a random hollow mask
        mask = torch.zeros(batch_size, max_seq_length, max_seq_length)
        mask[:, :, :max_seq_length // 2] = 1

        transformer = HollowTransformer(num_layers, num_heads, d_model, ff_hidden_dim, input_vocab_size, max_seq_length,output_vocab_size)
        output = transformer(input_data, mask)

        print(f"Input Data Shape {input_data.shape}")
        print(f"Output Data Shape {output.shape}")

if __name__=="__main__":
    unittest.main()