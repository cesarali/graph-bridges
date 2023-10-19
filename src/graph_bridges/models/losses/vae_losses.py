import torch
from torch import nn

# Loss function: reconstruction + KL divergence losses
def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.BCELoss(reduction='sum')(recon_x, x.view(-1, recon_x.shape[-1]))
    if mu is not None:
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    else:
        KLD = 0.
    return BCE + KLD