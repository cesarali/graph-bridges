import os
import unittest
import torch
import diffusers

from graph_bridges.models.temporal_networks.networks_tau import UNet
from diffusers import UNet2DModel, UNet1DModel
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from graph_bridges import data_path

class TestUnet(unittest.TestCase):

    def test_unet(self):
        batch_size = 23
        raw_dir = os.path.join(data_path, "raw")
        train_dataset = CIFAR10(root=raw_dir, train=True, transform=ToTensor())
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        databatch = next(train_dataloader.__iter__())

        UNet = UNet2DModel(in_channels=3,out_channels=3)
        times = torch.rand(batch_size)
        X_ = UNet(databatch[0],times)

if __name__=="__main__":
    unittest.main()
