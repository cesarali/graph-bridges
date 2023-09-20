import torch
from dataclasses import dataclass

from torchtyping import TensorType
from typing import Tuple,List,Union

from torch.distributions import Bernoulli
from graph_bridges.utils.spin_utils import bool_to_spins
from graph_bridges.models.backward_rates.ctdd_backward_rate import BackwardRate
from graph_bridges.configs.config_sb import SBConfig

class SteinSpinEstimator:
    """
    """
    def __init__(self,
                 config:SBConfig,
                 device:torch.device,
                 **kwargs):
        self.stein_epsilon = config.stein.stein_epsilon
        self.stein_sample_size = config.stein.stein_sample_size

        self.epsilon_distribution = Bernoulli(torch.full((config.data.D,),
                                                         self.stein_epsilon))
        self.device = device

    def set_device(self,device):
        self.device = device

    def stein_sample_per_sample_point(self,X,S,current_time=None):
        """
        For each binary vector in our data sample x,
        we need a sample of stein epsilon to perform the averages

        Args:
            X torch.Tensor(number_of_paths,number_of_spins)
            S torch.Tensor(number_of_sample,number_of_spins)
        """
        number_of_paths = X.shape[0]
        sample_size = S.shape[0]

        S_copy = S.repeat((number_of_paths, 1))
        X_copy = X.repeat_interleave(sample_size, 0)
        if current_time is not None:
            current_time = current_time.repeat_interleave(sample_size,0)
            return X_copy, S_copy, current_time
        else:
            return X_copy, S_copy

    def estimator(self,
                  phi:BackwardRate,
                  X:TensorType["batch_size", "dimension"],
                  current_time:TensorType["batch_size"]):
        """
        This function calculates $\phi(x_{\d},-x_d)$ i.e.
        for a data sample X torch.Tensor of size (number_of_paths,number_of_spins)

        we return

        Args
            X torch.Tensor(size=(batch_size,number_of_spins), {1.,-1.})

         :returns
            estimator torch.Tensor(size=batch_size*number_of_spins,1))
        """
        # HERE WE MAKE SURE THAT THE DATA IS IN SPINS
        if X.dtype == torch.bool:
            X = bool_to_spins(X)
        S = self.epsilon_distribution.sample(sample_shape=(self.stein_sample_size,)).bool().to(self.device)
        S = ~S  # Manfred's losses requieres epsilon as the probability for -1.
        S = bool_to_spins(S)

        number_of_paths = X.shape[0]
        number_of_spins = X.shape[1]

        # ESTIMATOR
        X_stein_copy,S_stein_copy,current_time = self.stein_sample_per_sample_point(X,S,current_time)
        stein_estimator = phi.stein_binary_forward(S_stein_copy * X_stein_copy,current_time)
        stein_estimator = (1. - S_stein_copy) * stein_estimator
        stein_estimator = stein_estimator.reshape(number_of_paths,
                                                  self.stein_sample_size,
                                                  number_of_spins)

        stein_estimator = stein_estimator.mean(axis=1)
        stein_estimator = (1 / (2. * self.stein_epsilon)) * stein_estimator
        stein_estimator = stein_estimator.reshape(number_of_paths * number_of_spins, 1)
        return stein_estimator

class BackwardRatioSteinEstimator:
    """
    """
    def __init__(self,
                 config:SBConfig,
                 device):
        self.dimension = config.loss.dimension_to_check
        self.stein_estimator = SteinSpinEstimator(config,device)
        self.device = device

    def set_device(self,
                   device):
        self.device = device
        self.stein_estimator.set_device(device)

    def estimator(self,
                  current_model: BackwardRate,
                  past_model: BackwardRate,
                  X_spins: TensorType["batch_size", "dimension"],
                  current_time: TensorType["batch_size"]):
        """
        :param current_model:
        :param X_spins:
        :return:
        """
        batch_size = X_spins.shape[0]
        number_of_spins = X_spins.shape[1]

        phi_new_d = current_model.stein_binary_forward(X_spins, current_time).squeeze()
        with torch.no_grad():
            phi_old_d = past_model.stein_binary_forward(X_spins, current_time)
            phi_old_d = phi_old_d.squeeze()

        # stein estimate
        stein_estimate = self.stein_estimator.estimator(current_model, X_spins, current_time)
        stein_estimate = stein_estimate.reshape(batch_size, number_of_spins)

        # stein estimate
        loss = (phi_new_d ** 2) - (2. * stein_estimate * phi_old_d)

        if self.dimension is not None:
            loss_d = loss[:,self.dimension]
            return loss_d.mean()
        else:
            return loss.mean()