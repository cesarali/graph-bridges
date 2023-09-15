import torch
from torch import nn
from ..ema import EMA
import torch.nn.functional as F

class MultiheadHollowAttention(torch.nn.Module):
    def __init__(self, num_heads, hidden_dim):
        super(MultiheadHollowAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = hidden_dim
        self.head_dim = hidden_dim // num_heads

        self.query_projection = torch.nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = torch.nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = torch.nn.Linear(hidden_dim, hidden_dim)

        self.final_projection = torch.nn.Linear(hidden_dim, hidden_dim)

    def split_heads(self, tensor, batch_size):
        return tensor.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query, key, value,mask):
        batch_size = query.shape[0]
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))

        # Apply hollow mask in both directions
        hollow_mask_forward = torch.triu(torch.ones(scores.shape[-2:]), diagonal=1)
        hollow_mask_backward = torch.tril(torch.ones(scores.shape[-2:]), diagonal=-1)

        hollow_mask_forward = hollow_mask_forward.unsqueeze(0).unsqueeze(1).to(query.device)
        hollow_mask_backward = hollow_mask_backward.unsqueeze(0).unsqueeze(1).to(query.device)

        scores.masked_fill_(hollow_mask_forward == 1, float('-inf'))
        scores.masked_fill_(hollow_mask_backward == 1, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attention_weights, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.final_projection(output)

        return output


class HollowTransformerLayer(nn.Module):

    def __init__(self, num_heads, hidden_dim, ff_hidden_dim, temb_dim=None):
        super(HollowTransformerLayer, self).__init__()
        self.multihead_attention = MultiheadHollowAttention(num_heads, hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, hidden_dim)
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        if temb_dim is not None:
            self.dense0 = nn.Linear(temb_dim, hidden_dim)
            self.act = nn.functional.silu

            nn.init.zeros_(self.dense0.bias)

    def forward(self, x, mask, temb=None):

        attention_output = self.multihead_attention(x, x, x, mask)
        x = self.layer_norm1(x + attention_output)
        
        if temb is not None:
            temb = self.dense0(self.act(temb))
            x += temb[:,None,:]

        feedforward_output = self.feedforward(x)
        x = self.layer_norm2(x + feedforward_output)

        return x

class HollowTransformer(nn.Module):

    def __init__(self, num_layers, num_heads, d_model, ff_hidden_dim, input_vocab_size, max_seq_length,
                 output_vocab_size):
        super(HollowTransformer, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.transformer_layers = nn.ModuleList(
            [HollowTransformerLayer(num_heads, d_model, ff_hidden_dim) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(d_model, output_vocab_size)

    def forward(self, x, mask):
        x = self.embedding(x)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, mask)
        x = self.fc(x)
        return x