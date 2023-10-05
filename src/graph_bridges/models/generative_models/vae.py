import os
import sys
from dataclasses import dataclass, asdict

# torch stuff
import torch
from torch import nn

# load models
from graph_bridges.data.dataloaders_utils import load_dataloader
from graph_bridges.models.networks_arquitectures.network_utils import load_encoder,load_decoder
#from ssda.trainers.trainers_utils import load_experiments_configuration

# configs
from graph_bridges.configs.config_vae import VAEConfig

# models
from graph_bridges.models.networks_arquitectures.decoder import Decoder
from graph_bridges.models.networks_arquitectures.encoder import Encoder


EPSILON = 1e-12

class VAE(nn.Module):

    def __init__(self,
                 config:VAEConfig=None,
                 experiment_name='ssda',
                 experiment_type='mnist',
                 experiment_indentifier="test",
                 checkpoint=None,
                 device=torch.device("cpu"),
                 read=False):
        super(VAE,self).__init__()

        self.config = config

        if self.config is not None:
            self.create_new_from_config()
        elif read:
            self.load_results_from_directory(experiment_name=experiment_name,
                                             experiment_type=experiment_type,
                                             experiment_indentifier=experiment_indentifier,
                                             checkpoint=checkpoint,
                                             device=device)

    def forward(self,image):
        z, mu, logvar = self.encoder(image)
        return self.decoder(z),mu,logvar

    def to(self,device):
        super().to(device)
        self.encoder.to(device)
        self.decoder.to(device)

    def generate(self, number_of_samples=64):
        # Generating samples from the trained VAE
        self.eval()
        with torch.no_grad():
            z_sample = torch.randn(number_of_samples, 20)
            sample = self.decoder(z_sample).cpu()
        return sample

    #===============================
    # files and stuff
    #===============================

    def create_new_from_config(self, config:VAEConfig, device=torch.device("cpu")):
        self.config = config
        self.config.initialize_new_experiment()

        self.dataloader = load_dataloader(self.config)

        self.encoder = load_encoder(self.config)
        self.encoder.to(device)

        self.decoder = load_decoder(self.config)
        self.decoder.to(device)

    def load_results_from_directory(self,
                                    experiment_name='ssda',
                                    experiment_type='mnist',
                                    experiment_indentifier="test",
                                    checkpoint=None,
                                    device=torch.device("cpu")):

        self.encoder,self.decoder, self.dataloader = load_experiments_configuration(experiment_name,
                                                                                    experiment_type,
                                                                                    experiment_indentifier,
                                                                                    checkpoint)
        self.encoder.to(device)
        self.decoder.to(device)


if __name__=="__main__":
    from graph_bridges.utils.plots.images_plots import plot_sample

    va = VAE()
    va.load_results_from_directory(experiment_name='ssda',
                                    experiment_type='mnist',
                                    experiment_indentifier="vae_train_example",
                                    checkpoint=None)
    sample_ = va.generate(number_of_samples=64)
    plot_sample(sample_)
