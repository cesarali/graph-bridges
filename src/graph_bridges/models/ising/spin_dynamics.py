import torch
from pprint import pprint
from torch.distributions.uniform import Uniform
from torch.distributions import Exponential, Bernoulli

from abc import ABC, abstractmethod
from discrete_diffusion.models.ising.spin_utils import spins_to_bool,bool_to_spins
from discrete_diffusion.models.ising.ising_parameters import ParametrizedIsingHamiltonian
from discrete_diffusion.models.stochastic_processes.birth_death_efficient import BirthDeathForwardBase, GaussianTargetRate
from torchtyping import patch_typeguard, TensorType
import math

def process_in_grid_to_times(paths,times_grid):
    batch_size, number_of_time_steps, number_of_spins = paths.shape[0],paths.shape[1],paths.shape[2]
    paths = paths.reshape(batch_size * number_of_time_steps, -1)
    times = times_grid[None, :].repeat((batch_size, 1)).unsqueeze(-1)
    times = times.reshape(batch_size * number_of_time_steps, -1)
    return paths,times

class ReferenceProcess(ABC):
    """
    Base class for reference process

    on_grid
    """
    name_="base_reference_process"
    def __init__(self,**kwargs):
        self.T = kwargs.get("T")
        self.tau = kwargs.get("tau")
        self.on_grid = kwargs.get("on_grid")
        self.time_grid = kwargs.get("time_grid")
        self.number_of_spins = kwargs.get("number_of_spins")

        self.reference_process_name = kwargs.get("reference_process_name")
        self.reference_process_parameters = kwargs
        self.rates_device = torch.device("cpu")

    def __call__(self,paths_batch,current_time):
        """
            It should define the transition rates for binary systems i.e. the times per unit
            time that spin in position x_d changes sign at time t.

            The children class should define
                forward_states_and_times

        :param paths_batch:
        :param current_time:
        :param training bool:
        :return:
        """
        if len(current_time.shape) == 0: # same time for all states
            return self.rates_states_and_time(paths_batch, current_time)
        elif len(current_time.shape) == 1: # time is on the grid
            assert len(paths_batch.shape) == 3
            return self.rates_path_and_grid(paths_batch, current_time)
        elif len(current_time.shape) == 2: # time per state
            return self.rates_states_and_times(paths_batch, current_time)
        elif len(current_time.shape) == 3: # time per state on a path
            return self.rates_paths_and_pathtimes(paths_batch, current_time)
        else:
            raise Exception("Time shape not proper in Ratio Estimator")

    def set_rates_device(self,rates_device):
        self.rates_device = rates_device

    def rates_states_and_time(self, paths_batch, current_time):
        """
        :param paths_batch:
        :param current_time: a
        :return:
        """
        assert len(paths_batch.shape) == 2
        time_as_vector = torch.full_like(paths_batch[:,0],current_time,device=self.rates_device)
        return self.rates_states_and_times(paths_batch, time_as_vector)

    def rates_path_and_grid(self, paths_batch, current_time):
        """

        :param paths_batch: torch.Tensor(size=(batch_size,number_of_time_steps,number_of_spins))
        :param current_time: torch.Tensor(size=(number_of_time_steps))
        :return:
        """
        batch_size = paths_batch.shape[0]
        number_of_time_steps = paths_batch.shape[1]
        number_of_spins = paths_batch.shape[2]

        full_times = current_time.repeat(batch_size, 1)

        times_per_state = full_times.reshape(batch_size * number_of_time_steps, -1)
        paths_as_state = paths_batch.reshape(batch_size * number_of_time_steps, -1)

        ratio_estimator = self.rates_states_and_times(paths_as_state, times_per_state)
        ratio_estimator = ratio_estimator.reshape(batch_size, number_of_time_steps, number_of_spins)

        return ratio_estimator

    def rates_paths_and_pathtimes(self, states, times):
        """
        :param states: torch.Tensor(size=(batch_size,number_of_time_steps,number_of_spins))
        :param times:  torch.Tensor(size=(batch_size,number_of_time_steps,1))
        """
        batch_size = states.shape[0]
        number_of_time_steps = states.shape[1]
        number_of_spins = states.shape[2]

        states = states.reshape(batch_size*number_of_time_steps,-1)
        times = times.reshape(batch_size*number_of_time_steps,-1)

        ratio_estimator = self.rates_states_and_times(states, times)
        ratio_estimator = ratio_estimator.reshape(batch_size,number_of_time_steps,number_of_spins)

        return ratio_estimator

    @abstractmethod
    def rates_to_device(self):
        return None

    @abstractmethod
    def rates_states_and_times(self, states, times):
        """
        :param states: torch.Tensor(size=(batch_size,number_of_spins))
        :param times:  torch.Tensor(size=(batch_size,1))
        :return: rates_states_and_times torch.Tensor(size=(batch_size,number_of_spins))
        """
        return None

    @abstractmethod
    def process_paths_on_times(self,start_spins,times):
        return None

    @abstractmethod
    def process_paths_on_grid(self,start_spins):
        return None

    def process_paths(self,start_spins,times=None,on_grid=True,one_step=False):
        if on_grid:
            paths, times = self.process_paths_on_grid(start_spins)
        else:
            if times is None:
                batch_size = start_spins.shape[0]
                time_distribution = Uniform(0., self.T)
                times = time_distribution.sample(sample_shape=(batch_size,))
            paths, times = self.process_paths_on_times(start_spins,times)
        return paths, times

    @abstractmethod
    def get_parameters(self):
        return None

    def obtain_parameters(self):
        return self.reference_process_parameters

    def target_distribution(self):
        return None

class GlauberProcess(ReferenceProcess):
    """
    Glauber Dynamics
    """
    name_="glauber_process"
    ising_model : ParametrizedIsingHamiltonian
    def __init__(self,**kwargs):
        super(GlauberProcess,self).__init__(**kwargs)
        self.mu = self.reference_process_parameters.get("mu")
        self.ising_model = self.reference_process_parameters.get("ising_model")
        self.ising_model_rates = torch.clone(self.ising_model)

    def process_paths_on_times(self, start_spins, times):
        paths, times_grid = self.process_paths_on_grid(start_spins)
        paths,times = process_in_grid_to_times(paths, times_grid)
        return paths,times

    def process_paths_on_grid(self,start_spins):
        if len(start_spins.shape) == 2:
            paths = start_spins.unsqueeze(1)
        elif len(start_spins.shape) == 3:
            paths = start_spins
            pass
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
            H_i = self.ising_model.hamiltonian_diagonal(states, i_random)
            x_i = torch.diag(states[:, i_random])
            flip_probability = (self.tau * self.mu * torch.exp(-x_i * H_i)) / 2 * torch.cosh(H_i)
            r = torch.rand((number_of_paths,))
            where_to_flip = r < flip_probability

            new_states = torch.clone(states)
            index_to_change = (rows_index[torch.where(where_to_flip)], i_random[torch.where(where_to_flip)])
            new_states[index_to_change] = states[index_to_change] * -1.

            paths = torch.cat([paths, new_states.unsqueeze(1)], dim=1)

        return paths, self.time_grid

    def rates_to_device(self):
        self.ising_model_rates.to(self.rates_device)

    def rates_states_and_times(self,states,times):
        batch_size = states.shape[0]
        number_of_spins = states.shape[-1]

        # SELECTS
        all_index = torch.arange(0, number_of_spins)
        all_index = all_index.repeat((batch_size))
        states = states.repeat_interleave(number_of_spins, 0)

        # EVALUATES HAMILTONIAN
        H_i = self.ising_model_rates.hamiltonian_diagonal(states, all_index)
        x_i = states[all_index, all_index]
        rate_per_spin = (self.mu * torch.exp(-x_i * H_i)) / 2 * torch.cosh(H_i)
        #flip_probability = (self.tau * self.mu * torch.exp(-x_i * H_i)) / 2 * torch.cosh(H_i)
        rate_per_spin = rate_per_spin.reshape(batch_size,number_of_spins)
        return rate_per_spin

    @classmethod
    def get_parameters(self):
        kwargs = {
            "reference_process_name":self.name_,
            "mu":1.,
            "ising_model":None,
        }
        return kwargs

class BinaryBirthDeath(ReferenceProcess):
    """
    """
    name_="binary_birth_death"
    def __init__(self,**kwargs):
        super(BinaryBirthDeath,self).__init__(**kwargs)
        self.thermostat_function_str = self.reference_process_parameters.get("thermostat_function")
        self.thermostat_parameters = self.reference_process_parameters.get("thermostat_function_parameters")

    def define_thermostat(self):
        if self.thermostat_function_str == "basic":
            self.thermostat_ = lambda t: torch.sin((torch.pi * .5) * t)
        else:
            raise Exception("Not Implemented")

    def process_paths_on_grid(self,start_spins):
        """
        :param paths:
        :param time_grid:
        :param thermostat_function:
        :param forward:
        :return:
        """
        batch_size = start_spins.shape[0]
        number_of_spins = start_spins.shape[1]

        paths = spins_to_bool(start_spins)
        paths = paths.unsqueeze(1)
        thermostat_values = torch.sigmoid(self.thermostat_function(self.time_grid))

        number_of_time_steps = self.time_grid.shape[0]
        time_indices = torch.arange(0, number_of_time_steps + 1, 1)
        time_direction = torch.clone(time_indices)

        if self.forward:
            time_direction_ = torch.clone(time_direction)
        else:
            time_direction_ = torch.flip(torch.clone(time_direction), dims=(0,))

        for i in range(1, number_of_time_steps):
            time_index = time_direction_[i]
            thermostat_value = thermostat_values[time_index]
            flip_probability = Bernoulli(probs=torch.Tensor([thermostat_value]))
            flips = flip_probability.sample(
                sample_shape=(batch_size, number_of_spins)).squeeze().bool()
            next_step = paths[:, i - 1, :] ^ flips
            paths = torch.concatenate([paths, next_step.unsqueeze(1)], axis=1)
        paths = bool_to_spins(paths)

        return paths

    def process_paths_on_times(self,start_spins,times):
        paths, times_grid = self.process_paths_on_grid(start_spins)
        paths, times = process_in_grid_to_times(paths, times_grid)
        return paths,times

    def rates_states_and_times(self,states,times):
        return None

    @classmethod
    def get_parameters(self):
        kwargs = {
            "thermostat_function":"basic",
            "thermostat_function_parameters":{}
        }
        return kwargs

    def target_distribution(self):
        return None

class EfficientDiffusion(ReferenceProcess):
    """
    """
    name_="efficient_diffusion"
    def __init__(self,**kwargs):
        super(EfficientDiffusion,self).__init__(**kwargs)
        self.reference_process_parameters = kwargs
        self.reference_process_name = kwargs.get("reference_process_name")
        self.target = self.reference_process_parameters.get("target")
        if self.target == "birth_death":
            self.model = BirthDeathForwardBase(**self.reference_process_parameters)
        elif self.target == "gaussian":
            self.model = GaussianTargetRate(**self.reference_process_parameters)

    def process_paths_on_grid(self,start_spins):
        batch_size = start_spins.shape[0]
        number_of_times_steps = self.time_grid.shape[0]
        times = self.time_grid[None, :].repeat((batch_size, 1))
        times = times.reshape(number_of_times_steps * batch_size)
        paths = start_spins[:, None, :].repeat((1, number_of_times_steps, 1))
        paths = paths.reshape(number_of_times_steps * batch_size, -1)
        paths, times = self.process_paths_on_times(paths,times)

        return paths.reshape(batch_size, number_of_times_steps, -1), \
               self.time_grid
    def process_paths_on_times(self,start_spins,times):
        assert len(start_spins.shape) == 2
        batch_size, number_of_spins = start_spins.shape

        # From Doucet Original Code
        qt0 = self.model.transition(times) # (B, S, S)

        # Flips
        flip_probabilities = qt0[:, 0, 1]
        flip_probabilities = flip_probabilities[:, None].repeat((1, number_of_spins))
        flips = Bernoulli(flip_probabilities).sample()
        flips = (-1.) ** flips
        flipped_spin = start_spins * flips

        return flipped_spin, times

    def rates_to_device(self):
        return None

    def rate(self,t):
        return self.model.rate(t)

    def integral_rate(self,t):
        return self.model.integral_rate(t)

    def rates_states_and_times(self,states,times):
        number_of_spins = states.shape[-1]
        rate = self.model.rate(times)  # (B, S, S)
        flip_rate = rate[:, 0, 1]
        flip_rate = flip_rate[:, None].repeat((1, number_of_spins))
        return flip_rate

    @classmethod
    def get_parameters(self):
        # reference_process_parameters = {
        #    "reference_process_name":"efficient_diffusion_model",
        #    "target":"birth_death",
        #    "S":2,
        #    "sigma_min": 1.,
        #    "sigma_max": 1.5,
        #    "device":torch.device("cpu")
        # }
        #self.T = kwargs.get("T")
        #self.tau = kwargs.get("tau")
        #self.on_grid = kwargs.get("on_grid")
        #self.time_grid = kwargs.get("time_grid")
        T = 1.
        tau =  0.01
        time_grid = torch.arange(0., T + tau, tau)

        paths_dataloader_options = {"T": 1.,
                                    "tau": 0.01,
                                    "on_grid": True,
                                    "time_grid": time_grid,
                                    "number_of_spins": 5}

        reference_process_parameters = {
            "reference_process_name": "efficient_diffusion_model",
            "target": "gaussian",
            "rate_sigma": 6.0,
            "S": 2,
            "Q_sigma": Q_sigma,
            "time_exponential": 100.0,
            "time_base": 3.0,
            "device": torch.device("cpu")
        }
        return reference_process_parameters

    def target_distribution(self):
        return None

class MarginalIsingProcess(ReferenceProcess):
    """
    """
    name_ = "marginal_ising"
    def __init__(self,**kwargs):
        super(MarginalIsingProcess,self).__init__(**kwargs)
        self.ising_model = kwargs.get("ising_model")
        self.dynamic_load = kwargs.get("dynamic_load")
        self.dynamic_load_parameters = kwargs.get("dynamic_load_parameters")

    def process_paths_on_grid(self,start_spins):
        return start_spins, self.time_grid

    def process_paths_on_times(self,start_spins,times):
        paths, time_grid = self.process_paths_on_grid(start_spins)
        paths, times = process_in_grid_to_times(paths, time_grid)
        return paths, times

    def rates_states_and_times(self):
        return None

    @classmethod
    def get_parameters(self):
        reference_process_parameters = {'path_process_name': 'marginal_ising',
                                        'dynamic_load': 'basic',
                                        'dynamic_load_parameters': None,
                                        'ising_model': None}
        return reference_process_parameters

    def obtain_parameters(self):
        return self.reference_process_parameters

    def target_distribution(self):
        return None

class ReferenceProcessFactory(object):
    models: dict
    def __init__(self):
        self._models = {"glauber_process":GlauberProcess,
                        "binary_birth_death":BinaryBirthDeath,
                        "efficient_diffusion":EfficientDiffusion,
                        "marginal_ising":MarginalIsingProcess}

    def create(self, model_type: str, **kwargs):
        builder = self._models.get(model_type)
        if not builder:
            raise ValueError(f"Unknown recognition model {model_type}")
        return builder(**kwargs)


if __name__=="__main__":
    from discrete_diffusion.data.spins_dataloaders import BernoulliSpinsDataLoader

    number_of_spins = 5
    bernoulli_dataloader_parameters_0 = {'batch_size': 32,
                                         'bernoulli_probability': 0.5,
                                         'number_of_paths': 500,
                                         'number_of_spins': 10,
                                         'remove': True}
    bernoulli_dataloader_0 = BernoulliSpinsDataLoader(**bernoulli_dataloader_parameters_0)

    # ==============================================================================
    #  Efficient Birth Death
    #  here we have a configuration for a one sinkhorn step of diffusion
    # ==============================================================================

    T = 1.
    tau = 0.01
    time_grid = torch.arange(0., T + tau, tau)

    paths_dataloader_options = {"T": 1.,
                                "tau": 0.01,
                                "on_grid": True,
                                "time_grid": time_grid,
                                "number_of_spins": 5}

    Q_sigma = 512.
    reference_process_parameters = {
        "reference_process_name":"efficient_diffusion",
        "target":"gaussian",
        "rate_sigma": 6.0,
        "S":2,
        "Q_sigma": Q_sigma,
        "time_exponential": 100.0,
        "time_base": 3.0,
        "device":torch.device("cpu")
    }

    reference_process_parameters.update(paths_dataloader_options)

    # =====================================
    # PROCESS
    # =====================================

    reference_process_name = reference_process_parameters.get("reference_process_name")

    reference_process_factory = ReferenceProcessFactory()
    reference_process = reference_process_factory.create(reference_process_name,
                                                         **reference_process_parameters)
    spins_0 = next(bernoulli_dataloader_0.train().__iter__())[0]

    paths, times = reference_process.process_paths(spins_0)

    integral_ = reference_process.integral_rate(times)

    # ==============================================================================
    #  TEST PATHS DATA LOADERS
    # ==============================================================================
    """
    from discrete_diffusion.data.paths_dataloaders import PathsDataloader
    from discrete_diffusion.data.spins_dataloaders import IsingSpinsDataLoader
    from discrete_diffusion.data.spins_dataloaders import BernoulliSpinsDataLoader
    from discrete_diffusion.data.spins_dataloaders import EfficientDiffusionDataLoader

    Q_sigma = 512.
    batch_size = 32
    number_of_spins = 3

    ising_dataloader_parameters = IsingSpinsDataLoader.get_parameters()
    ising_dataloader_parameters.update({"remove":False,
                                        "batch_size":batch_size,
                                        "number_of_spins":number_of_spins})
    ising_dataloader = IsingSpinsDataLoader(**ising_dataloader_parameters)

    efficient_dataloader_parameters = EfficientDiffusionDataLoader.get_parameters()
    efficient_dataloader_parameters.update({"number_of_spins":number_of_spins,
                                            "Q_sigma":Q_sigma,
                                            "remove":False,
                                            "batch_size":batch_size})

    efficient_dataloader = EfficientDiffusionDataLoader(**efficient_dataloader_parameters)
    spins_dataloader_0 = ising_dataloader
    spins_dataloader_1 = efficient_dataloader
    """
    # ==============================================================================
    #  Efficient Birth Death
    #  here we have a configuration for a one sinkhorn step of diffusion
    # ==============================================================================
    """
    #reference_process_parameters = {
    #    "reference_process_name":"efficient_diffusion",
    #    "target":"birth_death",
    #    "S":2,
    #    "sigma_min": 1.,
    #    "sigma_max": 1.5,
    #    "device":torch.device("cpu")
    #}

    reference_process_parameters = {
        "reference_process_name":"efficient_diffusion",
        "target":"gaussian",
        "rate_sigma": 6.0,
        "S":2,
        "Q_sigma": Q_sigma,
        "time_exponential": 100.0,
        "time_base": 3.0,
        "device":torch.device("cpu")
    }

    paths_dataloader_parameters = {'tau': 0.025,
                                   'T': 1.0,
                                   'on_grid':True,
                                   'parametric_backward_process': 'poisson_naive',
                                   'reference_process_parameters': reference_process_parameters}

    paths_dataloader = PathsDataloader(spins_dataloader_0,
                                       spins_dataloader_1,
                                       **paths_dataloader_parameters)

    paths, times = next(paths_dataloader.train().__iter__())
    print(paths.shape)
    print(times.shape)

    states = paths[:,10,:]
    times_ = Uniform(0.,paths_dataloader.T).sample(sample_shape=(batch_size,)).unsqueeze(-1)
    rates_per_spin = paths_dataloader.reference_process(states,times_)
    print(rates_per_spin.shape)
    """
    # ==================================================
    # Binary Birth Death
    # ==================================================
    """
    spins_dataloader_0 = ising_dataloader
    spins_dataloader_1 = bernoulli_dataloader
    # PATHS LOADERS
    paths_dataloader_parameters = PathsDataloader.get_parameters()
    pprint(paths_dataloader_parameters)

    paths_dataloader_parameters.update({"tau": 0.01})
    paths_process_parameters = paths_dataloader_parameters.get("paths_process_parameters")
    paths_process_parameters.update({"ising_model": None,
                                     "path_process_name": "binary_birth_death"})

    paths_dataloader = PathsDataloader(spins_dataloader_0, **paths_dataloader_parameters)
    paths_batch = next(paths_dataloader.train().__iter__())
    print(paths_batch.shape)
    """
    # ==================================================
    # Parametrized Ising for Glauber Dynamics
    # ==================================================

    """
    ising_parameters = ParametrizedIsingHamiltonian.get_parameters()
    number_of_couplings = ParametrizedIsingHamiltonian.coupling_size(number_of_spins)

    # fields = torch.full((number_of_spins,),0.2)
    # couplings = torch.full((number_of_couplings,),.1)
    fields = torch.Tensor(size=(number_of_spins,)).normal_(0., 1. / number_of_spins)
    couplings = torch.Tensor(size=(number_of_couplings,)).normal_(0., 1 / number_of_spins)

    ising_parameters.update({"number_of_spins": number_of_spins,
                             'couplings_deterministic': None,
                             'obtain_partition_function': False,
                             "number_of_mcmc_steps": 1000,
                             'couplings_sigma': None,
                             "fields": fields,
                             "couplings": couplings})
    ising_model_for_glauber = ParametrizedIsingHamiltonian(**ising_parameters)

    # PATHS LOADERS GLAUBER
    reference_process_parameters = {
        "reference_process_name": "glauber_process",
        "mu": 1.,
        "ising_model": ising_model_for_glauber,
    }

    paths_dataloader_parameters = {'tau': 0.025,
                                   'T': 1.0,
                                   'on_grid':True,
                                   'parametric_backward_process': 'poisson_naive',
                                   'reference_process_parameters': reference_process_parameters}

    paths_dataloader = PathsDataloader(spins_dataloader_0,
                                       spins_dataloader_1,
                                       **paths_dataloader_parameters)

    paths_batch,times = next(paths_dataloader.train().__iter__())
    print("Shapes for glauber as reference")
    print(paths_batch.shape)
    print(times.shape)

    states = paths_batch[:,10,:]
    times_ = Uniform(0.,paths_dataloader.T).sample(sample_shape=(batch_size,)).unsqueeze(-1)
    rates_per_spin = paths_dataloader.reference_process(states,times_)
    print(rates_per_spin.shape)
    """
    # ==================================================
    # Marginal Ising
    # ==================================================
    """
    from discrete_diffusion.data.paths_dataloaders import MarginalIsingDataLoader
    reference_process_parameters = {'reference_process_name': 'marginal_ising',
                                    'dynamic_load': 'basic',
                                    'dynamic_load_parameters': None,
                                    'ising_model': None}

    marginal_ising_dataloader_parameters = {'number_of_spins': number_of_spins,
                                            'number_of_paths': 100,
                                            'batch_size': batch_size,
                                            'data_path': None,
                                            'remove': False,  # If True a new data set is created
                                            'tau': 0.025,
                                            # Time delta for grid spacing in simulations depending on Poisson simulations
                                            'T': 1.0,  # Final time of the process
                                            'reference_process_parameters': reference_process_parameters}

    marginal_ising_dataloader = MarginalIsingDataLoader(**marginal_ising_dataloader_parameters)
    spins_dataloader_0 = marginal_ising_dataloader  # data
    spins_dataloader_1 = marginal_ising_dataloader  # target

    paths_dataloader_parameters = {'tau': 0.025,
                                   'T': 1.0,
                                   'on_grid':False,
                                   'parametric_backward_process': 'poisson_naive',
                                   'reference_process_parameters': reference_process_parameters}

    paths_dataloader = PathsDataloader(spins_dataloader_0,
                                       spins_dataloader_1,
                                       **paths_dataloader_parameters)

    paths, times = next(paths_dataloader.train().__iter__())
    print(paths.shape)
    print(times.shape)
    """