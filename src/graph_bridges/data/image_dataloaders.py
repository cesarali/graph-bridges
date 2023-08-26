import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.datasets
import torchvision.transforms
import os
from graph_bridges.data.image_dataloader_config import DiscreteCIFAR10Config
from graph_bridges.configs.images.config_ctdd import CTDDConfig
class DiscreteCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, data_root,train=True,download=True,random_flips=False):
        super().__init__(root=data_root,
                         train=train,
                         download=download)

        self.data = torch.from_numpy(self.data)
        self.data = self.data.transpose(1,3)
        self.data = self.data.transpose(2,3)

        self.targets = torch.from_numpy(np.array(self.targets))

        # Put both data and targets on GPU in advance
        self.data = self.data.view(-1, 3, 32, 32)

        self.random_flips = random_flips
        if self.random_flips:
            self.flip = torchvision.transforms.RandomHorizontalFlip()

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, 'CIFAR10', 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, 'CIFAR10', 'processed')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.random_flips:
            img = self.flip(img)

        return img

class DiscreteCIFAR10Dataloader():
    """

    """

    def __init__(self,cfg:CTDDConfig,device=torch.device("cpu")):
        train_dataset = DiscreteCIFAR10(data_root=cfg.data.dir,train=True)
        test_dataset =  DiscreteCIFAR10(data_root=cfg.data.dir,train=True)

        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                            batch_size=cfg.data.batch_size,
                                                            shuffle=True)

        self.test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                           batch_size=cfg.data.batch_size,
                                                           shuffle=True)
    def train(self):
        return self.train_dataloader

    def test(self):
        return self.test_dataloader

class LakhPianoroll(Dataset):
    def __init__(self, cfg, device):
        S = cfg.data.S
        L = cfg.data.shape[0]
        np_data = np.load(cfg.data.path) # (N, L) in range [0, S)

        self.data = torch.from_numpy(np_data).to(device)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        return self.data[index]