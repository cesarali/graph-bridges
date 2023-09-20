import os
import torch
import pickle

from pathlib import Path
from typing import Union,List

from dataclasses import dataclass, asdict

from torchvision import transforms
from torch.utils.data import TensorDataset
from graph_bridges.configs.config_sb import SBConfig
from graph_bridges.configs.config_ctdd import CTDDConfig
from graph_bridges.data.ising_dataloaders_config import ParametrizedIsingHamiltonianConfig
from graph_bridges.models.spin_glass.ising_parameters import ParametrizedIsingHamiltonian

from graph_bridges.data.transforms import (
    FlattenTransform,
    UnsqueezeTensorTransform,
    SqueezeTransform,
    UnFlattenTransform,
    FromUpperDiagonalTransform,
    ToUpperDiagonalIndicesTransform,
    BinaryTensorToSpinsTransform,
    SpinsToBinaryTensor
)

def get_spins_transforms(config:ParametrizedIsingHamiltonianConfig):
    """
    :param config:

    :return: transform_list,inverse_transform_list
    """
    transform_list  = []
    inverse_transform_list = []
    if config.as_image:
        transform_list.extend([UnsqueezeTensorTransform(1),UnsqueezeTensorTransform(1)])
        inverse_transform_list.extend([SqueezeTransform])

    if config.doucet:
        assert not config.as_spins
        transform_list.append(SpinsToBinaryTensor())
        inverse_transform_list.append(BinaryTensorToSpinsTransform)

    return transform_list,inverse_transform_list

@dataclass
class SpinGlassSimulationData:
    sample:List = None
    couplings:List = None
    fields:List = None
    beta:float = 1.

def simulate_spin_glass_data(config:ParametrizedIsingHamiltonianConfig)->SpinGlassSimulationData:
    number_of_spins = config.number_of_spins
    number_of_couplings = ParametrizedIsingHamiltonian.coupling_size(number_of_spins)

    if config.fields is None and config.couplings is None:
        #fields = torch.full((number_of_spins,),0.2)
        #couplings = torch.full((number_of_couplings,),.1)

        fields = torch.Tensor(size=(number_of_spins,)).normal_(0.,1./number_of_spins)
        couplings = torch.Tensor(size=(number_of_couplings,)).normal_(0.,1/number_of_spins)

        config.fields = fields.tolist()
        config.couplings = couplings.tolist()

    ising_model_real = ParametrizedIsingHamiltonian(config)
    ising_mcmc_sample = ising_model_real.sample(config)

    assert config.number_of_mcmc_burning_steps < config.number_of_mcmc_steps

    ising_sample = ising_mcmc_sample[:, config.number_of_mcmc_burning_steps,:]
    simulation_data = SpinGlassSimulationData(sample=ising_sample.tolist(),
                                              couplings=couplings.tolist(),
                                              fields=fields.tolist())
    return simulation_data
def get_ising_dataset(data_config:ParametrizedIsingHamiltonianConfig):
    data_config: ParametrizedIsingHamiltonianConfig

    data_ = data_config.data
    dataloader_data_dir = data_config.dataloader_data_dir
    dataloader_data_path = Path(data_config.dataloader_data_path)

    #==============================================
    # READ OR SIMULATE
    #==============================================
    if dataloader_data_path.exists():
        with open(dataloader_data_path, 'rb') as f:
            spin_glass_simulation = pickle.load(f)
            if data_config.couplings != spin_glass_simulation.couplings:
                print("Coupling from Spin File Different from Config")
            if data_config.fields != spin_glass_simulation.fields:
                print("Coupling from Spin File Different from Config")
    else:
        spin_glass_simulation = simulate_spin_glass_data(data_config)
        if not Path(dataloader_data_dir).exists():
            os.makedirs(dataloader_data_dir)
        pickle.dump(spin_glass_simulation,open(dataloader_data_path,"wb"))

    data_config.couplings = spin_glass_simulation.couplings
    data_config.fields = spin_glass_simulation.fields
    test_size = int(data_config.test_split * len(spin_glass_simulation.sample))
    train_spin_list, test_spin_list = spin_glass_simulation.sample[test_size:], spin_glass_simulation.sample[:test_size]
    train_spin = torch.Tensor(train_spin_list)
    test_spin = torch.Tensor(test_spin_list)

    return test_spin,train_spin

class ParametrizedIsingHamiltonianLoader:

    name_ = "ParametrizedIsingHamiltonianLoader"

    def __init__(self, config:ParametrizedIsingHamiltonianConfig,device):
        self.config = config

        self.batch_size = config.batch_size
        self.delete_data = config.delete_data

        self.doucet = config.doucet
        self.number_of_spins = config.D

        self.dataloader_data_dir = config.dataloader_data_dir
        self.dataloader_data_dir_path = Path(self.dataloader_data_dir)
        self.dataloader_data_dir_file_path = Path(config.dataloader_data_path)

        # transforms
        transform_list,inverse_transform_list = get_spins_transforms(self.config)
        self.composed_transform = transforms.Compose(transform_list)
        self.transform_to_spins = transforms.Compose(inverse_transform_list)

        # datasets
        train_dataset,test_dataset = get_ising_dataset(self.config)
        train_dataset = self.composed_transform(train_dataset)
        test_dataset = self.composed_transform(test_dataset)

        self.train_loader = torch.utils.data.DataLoader(
            TensorDataset(train_dataset),
            batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            TensorDataset(test_dataset),
            batch_size=self.batch_size, shuffle=True)

    def train(self):
        return self.train_loader

    def test(self):
        return self.test_loader
