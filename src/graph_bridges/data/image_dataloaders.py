import torch
from pathlib import Path

from torch.utils.data import Dataset
import numpy as np
import torchvision.datasets
import torchvision.transforms
import os
from graph_bridges.data.image_dataloader_config import DiscreteCIFAR10Config
from graph_bridges.configs.images.cifar10_config_ctdd import CTDDConfig
from graph_bridges.data.image_dataloader_config import NISTLoaderConfig
from torchvision import transforms,datasets

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

        return img,target

class DiscreteCIFAR10Dataloader():
    """

    """

    def __init__(self,cfg:CTDDConfig,device=torch.device("cpu")):
        train_dataset = DiscreteCIFAR10(data_root=cfg.data.dir,train=True)
        test_dataset =  DiscreteCIFAR10(data_root=cfg.data.dir,train=False)


        self.doucet = cfg.data.doucet
        self.number_of_spins = cfg.data.D

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

def get_data(config:NISTLoaderConfig,type="data"):
    if type=="data":
        data_config = config.data

    else:
        data_config = config.target

    data_ = data_config.data
    dataloader_data_dir = data_config.dataloader_data_dir
    batch_size = data_config.batch_size
    threshold = data_config.pepper_threshold

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: (x > threshold).float())])

    # Load MNIST dataset
    if data_ == "mnist":
        train_dataset = datasets.MNIST(dataloader_data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(dataloader_data_dir, train=False, download=True, transform=transform)
    elif data_ == "emnist":
        train_dataset = datasets.EMNIST(root=dataloader_data_dir,
                                        split='letters',
                                        train=True,
                                        download=True,
                                        transform=transform)

        test_dataset = datasets.EMNIST(root=dataloader_data_dir,
                                       split='letters',
                                       train=False,
                                       download=True,
                                       transform=transform)
    elif data_== "fashion":
        train_dataset = datasets.FashionMNIST(root=dataloader_data_dir,
                                              train=True,
                                              download=True,
                                              transform=transform)
        test_dataset = datasets.FashionMNIST(root=dataloader_data_dir,
                                             train=False,
                                             download=True,
                                             transform=transform)
    else:
        raise Exception("Data Loader Not Found!")

    data_config.training_size = len(train_dataset)
    data_config.test_size = len(test_dataset)
    data_config.total_data_size = data_config.training_size + data_config.test_size

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=True)


    return train_loader,test_loader

class NISTLoader:

    name_ = "NISTLoader"

    def __init__(self, config:NISTLoaderConfig,device):
        self.config = config

        self.batch_size = config.data.batch_size
        self.delete_data = config.data.delete_data

        self.doucet = config.data.doucet
        self.number_of_spins = config.data.D

        self.dataloader_data_dir = config.data.dataloader_data_dir
        self.dataloader_data_dir_path = Path(self.dataloader_data_dir)
        self.dataloader_data_dir_file_path = Path(config.data.dataloader_data_dir_file)

        self.train_loader,self.test_loader = get_data(self.config)

    def train(self):
        return self.train_loader

    def test(self):
        return self.test_loader

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