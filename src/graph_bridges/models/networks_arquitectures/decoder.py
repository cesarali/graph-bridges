import torch
from torch import nn
from ssda.configs.vae_config import VAEConfig

class Decoder(nn.Module):
    def __init__(self,config:VAEConfig):
        super(Decoder, self).__init__()

        self.fc3 = nn.Linear(config.z_dim, config.decoder.decoder_hidden_size)
        self.fc4 = nn.Linear(config.decoder.decoder_hidden_size, config.dataloader.input_dim)

    def forward(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))


