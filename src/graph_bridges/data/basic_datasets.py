import torch
from torch import matmul as m
from torch.utils.data import DataLoader, Dataset
from torch.distributions import Bernoulli, Normal
from torch.distributions import MultivariateNormal


# Define the dataset
class BasicDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DictDataSet(Dataset):
    """
    # Define your data dictionary
    data_dict = {'input': torch.randn(2, 10), 'target': torch.randn(2, 5)}

    # Create your dataset
    my_dataset = DictDataSet(data_dict)

    # Create a DataLoader from your dataset
    batch_size = 2
    dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
    """
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())

    def __len__(self):
        return len(self.data_dict[self.keys[0]])

    def __getitem__(self, idx):
        return {key: self.data_dict[key][idx] for key in self.keys}