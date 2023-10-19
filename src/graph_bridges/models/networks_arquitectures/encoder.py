import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

from graph_bridges.configs.config_vae import VAEConfig

# Define the VAE model
class Encoder(nn.Module):
    def __init__(self,config:VAEConfig):
        super(Encoder, self).__init__()
        self.config = config
        self.stochastic = self.config.encoder.stochastics
        self.fc1 = nn.Linear(self.config.dataloader.input_dim, self.config.encoder.encoder_hidden_size)
        self.fc21 = nn.Linear(self.config.encoder.encoder_hidden_size, self.config.z_dim)  # Mean
        if self.stochastic:
            self.fc22 = nn.Linear(self.config.encoder.encoder_hidden_size, self.config.z_dim)  # Variance


    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        mu = self.fc21(h1)
        if self.stochastic:
            logvar = self.fc22(h1)
        else:
            logvar = None
        return mu,logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        if len(x.shape) == 4:
            B,C,W,H = x.shape
            D = C*W*H
        elif len(x.shape) == 3:
            B, W, H = x.shape
            D = W*H
        mu, logvar = self.encode(x.view(-1, D))
        if self.stochastic:
            z = self.reparameterize(mu, logvar)
            return z, mu, logvar
        else:
            return mu,None,None
