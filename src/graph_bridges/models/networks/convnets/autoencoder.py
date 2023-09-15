import torch
from torch import nn
from dataclasses import dataclass, asdict
from graph_bridges.models.networks.embedding_utils import transformer_timestep_embedding



@dataclass
class ConvNetAutoencoderConfig:
    temp_name: str = "ConvNetAutoencoder"

    encoder_channels: int = 16
    latent_dim: int = 32
    decoder_channels: int = 16

    time_embed_dim : int = 128
    time_scale_factor :int = 1000


# Define a ResNet block
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_dim=None):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_dim is not None:
            self.act = nn.functional.silu
            self.dense0 = nn.Linear(temb_dim, out_channels)
            nn.init.zeros_(self.dense0.bias)

    def forward(self, x, temb=None):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x += residual  # Add the residual connection
        #x = self.relu(x)

        if temb is not None:
            h = self.dense0(self.act(temb))
            x+= h[:,:,None,None]
        return x


# Define the Autoencoder class
class ConvNetAutoencoder(nn.Module):
    def __init__(self, config,device):
        super(ConvNetAutoencoder, self).__init__()

        self.device = device

        self.in_channels = config.data.C

        self.encoder_channels = config.temp_network.encoder_channels
        self.latent_dim = config.temp_network.latent_dim
        self.decoder_channels = config.temp_network.encoder_channels

        self.time_embed_dim = config.temp_network.time_embed_dim
        self.time_scale_factor = config.temp_network.time_scale_factor

        self.do_time_embed = True
        self.act = nn.functional.silu

        # time
        self.temb_modules = []
        self.temb_modules.append(nn.Linear(self.time_embed_dim, self.time_embed_dim*4))
        nn.init.zeros_(self.temb_modules[-1].bias)
        self.temb_modules.append(nn.Linear(self.time_embed_dim*4, self.time_embed_dim*4))
        nn.init.zeros_(self.temb_modules[-1].bias)
        self.temb_modules = nn.ModuleList(self.temb_modules).to(self.device)
        self.expanded_time_dim = 4 * self.time_embed_dim if self.do_time_embed else None

        # Encoder
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.resnet_1 = ResNetBlock(32, 32, self.expanded_time_dim)  # Use the ResNet block

        self.encoder_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResNetBlock(64, 64,self.expanded_time_dim),  # Use the ResNet block
        )

        # Decoder
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.resnet_2 = ResNetBlock(32, 32,self.expanded_time_dim)  # Use the ResNet block

        self.decoder_2 = nn.Sequential(
            nn.Conv2d(32, 2*self.in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        self.to(self.device)

    def _time_embedding(self, timesteps):
        if self.do_time_embed:
            temb = transformer_timestep_embedding(
                timesteps * self.time_scale_factor, self.time_embed_dim
            )
            temb = self.temb_modules[0](temb)
            temb = self.temb_modules[1](self.act(temb))
        else:
            temb = None
        return temb

    def forward(self,x,timesteps):
        temp = self._time_embedding(timesteps)

        # Encoder
        x = self.encoder_1(x)
        x = self.resnet_1(x,temp)
        x = self.encoder_2(x)

        # Decoder
        x = self.decoder_1(x)
        x = self.resnet_2(x,temp)
        x = self.decoder_2(x)

        return x
