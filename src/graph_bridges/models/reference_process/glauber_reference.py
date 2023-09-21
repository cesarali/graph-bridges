import os
import math
import torch
import numpy as np
from .ctdd_reference import ReferenceProcess
from torchtyping import TensorType

from graph_bridges.configs.spin_glass.spin_glass_config_sb import SBConfig
from graph_bridges.models.spin_glass.ising_parameters import ParametrizedSpinGlassHamiltonian
from graph_bridges.data.spin_glass_dataloaders_config import ParametrizedSpinGlassHamiltonianConfig
from graph_bridges.models.reference_process.reference_process_config import GlauberDynamicsConfig

class GlauberDynamics(ReferenceProcess):

    def __init__(self, cfg:SBConfig, device,rank=None):
        ReferenceProcess.__init__(self,cfg,device)

        self.D = cfg.data.D
        self.S = cfg.data.S
        self.gamma = cfg.reference.gamma
        self.beta = cfg.reference.beta
        self.min_t = cfg.sampler.min_t
        self.tau = self.min_t
        self.device = device

        # create parametrized hamiltonian to be used in the computation of the dynamics
        if cfg.reference.fom_data_hamiltonian:
            assert isinstance(cfg.data, ParametrizedSpinGlassHamiltonianConfig)
            self.hamiltonian = ParametrizedSpinGlassHamiltonian(cfg.data, self.device)
        else:
            if cfg.reference.fields is not None and cfg.reference.couplings is not None:
                assert isinstance(cfg.reference,GlauberDynamicsConfig)
                self.hamiltonian = ParametrizedSpinGlassHamiltonian(cfg.reference, self.device)
                cfg.reference.fields = self.hamiltonian.fields.tolist()
                cfg.reference.couplings = self.hamiltonian.couplings.tolist()

    def to(self,device):
        self.device = device
        self.hamiltonian.to(self.device)
        return self

    def _integral_rate_scalar(self, t: TensorType["B"]
                              ) -> TensorType["B"]:
        return None
    def _rate_scalar(self, t: TensorType["B"]
                     ) -> TensorType["B"]:
        return None

    def rate(self, t: TensorType["B"]
             ) -> TensorType["B", "S", "S"]:

        return None

    def transition(self, t: TensorType["B"]
                   ) -> TensorType["B", "S", "S"]:
        t = t.to(self.device)

        return None

    #============================================================
    # OLD FUNCTIONS
    #============================================================

    def rates_states_and_times(self,states,times):
        batch_size = states.shape[0]
        number_of_spins = states.shape[-1]

        # SELECTS
        all_index = torch.arange(0, number_of_spins)
        all_index = all_index.repeat((batch_size))
        states = states.repeat_interleave(number_of_spins, 0)

        # EVALUATES HAMILTONIAN
        H_i = self.hamiltonian.hamiltonian_diagonal(states, all_index)
        x_i = states[all_index, all_index]
        rate_per_spin = (self.gamma * torch.exp(-x_i * H_i)) / 2 * torch.cosh(H_i)
        #flip_probability = (self.tau * self.mu * torch.exp(-x_i * H_i)) / 2 * torch.cosh(H_i)
        rate_per_spin = rate_per_spin.reshape(batch_size,number_of_spins)
        return rate_per_spin

    def glauber_dynamics(self,start_spins,time_steps):
        if len(start_spins.shape) == 2:
            paths = start_spins.unsqueeze(1)
        elif len(start_spins.shape) == 3:
            paths = start_spins
        else:
            print("Wrong Path From Initial Distribution, Dynamic not possible")
            raise Exception

        number_of_paths = paths.shape[0]
        number_of_spins = paths.shape[-1]
        rows_index = torch.arange(0, number_of_paths)
        for time_index in self.time_grid[1:]:
            states = paths[:, -1, :]

            i_random = torch.randint(0, number_of_spins, (number_of_paths,))

            # EVALUATES HAMILTONIAN
            H_i = self.hamiltonian.hamiltonian_diagonal(states, i_random)
            x_i = torch.diag(states[:, i_random])
            flip_probability = (self.tau * self.mu * torch.exp(-x_i * H_i)) / 2 * torch.cosh(H_i)
            r = torch.rand((number_of_paths,))
            where_to_flip = r < flip_probability

            new_states = torch.clone(states)
            index_to_change = (rows_index[torch.where(where_to_flip)], i_random[torch.where(where_to_flip)])
            new_states[index_to_change] = states[index_to_change] * -1.

            paths = torch.cat([paths, new_states.unsqueeze(1)], dim=1)

        return paths, self.time_grid