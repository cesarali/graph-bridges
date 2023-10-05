import torch
from dataclasses import dataclass

from torchtyping import TensorType
from typing import Tuple,List,Union

from torch.distributions import Bernoulli
from graph_bridges.configs.config_sb import SBConfig

from graph_bridges.utils.spin_utils import bool_to_spins
from graph_bridges.models.backward_rates.sb_backward_rate import SchrodingerBridgeBackwardRate
from graph_bridges.models.utils.jacobians import compute_jacobian_with_fix_time
from graph_bridges.models.losses.loss_configs import (
    GradientEstimatorConfig,
    SteinSpinEstimatorConfig,
    RealFlipConfig,
)

from graph_bridges.data.transforms import BinaryTensorToSpinsTransform
from graph_bridges.data.transforms import SpinsToBinaryTensor
from graph_bridges.models.spin_glass.spin_utils import flip_and_copy_spins
from graph_bridges.models.backward_rates.sb_backward_rate import SchrodingerBridgeBackwardRate

class RealFlip:
    """

    """
    def __init__(self,config=None,device=None):
        self.device = device

    def __call__(self,phi,X_spins,current_time)->TensorType["batch_size","number_of_spins"]:
        batch_size = X_spins.size(0)
        number_of_spins = X_spins.size(1)
        X_copy, X_flipped = flip_and_copy_spins(X_spins)
        copy_time = torch.repeat_interleave(current_time, X_spins.size(1))
        transition_rates_ = phi(X_flipped, copy_time)
        transition_rates = transition_rates_.reshape(batch_size, number_of_spins, number_of_spins)
        transition_rates = torch.einsum("bii->bi", transition_rates)
        return transition_rates

    def set_device(self,device):
        self.device = device
class GradientFlipEstimator:
    """
    """
    spins_to_binary = SpinsToBinaryTensor()

    def __init__(self,config,device):
        pass
    def __call__(self,
                 phi:SchrodingerBridgeBackwardRate,
                 X_binary:TensorType["batch_size", "dimension"],
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
        J = compute_jacobian_with_fix_time(phi,X_binary,current_time)
        J = torch.einsum("bii->bi",J)
        J = J.to(X_binary.device)
        X_binary = self.spins_to_binary(X_binary)
        flip = -(2. * X_binary - 1.) * J
        return flip

class SteinSpinEstimator:
    """
    This function calculates $\phi(x_{\d},-x_d)$ per dimension i.e.
    the flip rate in the given dimension if one flips in this dimension

    Args
        X torch.Tensor(size=(batch_size,number_of_spins), {1.,-1.})

     :returns
        estimator torch.Tensor(size=batch_size,number_of_spins))
    """
    def __init__(self,
                 config:SBConfig,
                 device:torch.device,
                 **kwargs):
        self.stein_epsilon = config.flip_estimator.stein_epsilon
        self.stein_sample_size = config.flip_estimator.stein_sample_size

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

    def __call__(self,
                 phi:SchrodingerBridgeBackwardRate,
                 X:TensorType["batch_size", "dimension"],
                 current_time:TensorType["batch_size"]):
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
        stein_estimator = phi.flip_rate(S_stein_copy * X_stein_copy, current_time)
        stein_estimator = (1. - S_stein_copy) * stein_estimator
        stein_estimator = stein_estimator.reshape(number_of_paths,
                                                  self.stein_sample_size,
                                                  number_of_spins)

        stein_estimator = stein_estimator.mean(axis=1)
        stein_estimator = (1 / (2. * self.stein_epsilon)) * stein_estimator
        return stein_estimator

class BackwardRatioSteinEstimator:
    """
    """
    def __init__(self,
                 config:SBConfig,
                 device):
        self.dimension = config.loss.dimension_to_check

        self.flip_old_time = config.loss.flip_old_time
        self.flip_current_time =  config.loss.flip_current_time

        if isinstance(config.flip_estimator,SteinSpinEstimatorConfig):
            self.flip_estimator = SteinSpinEstimator(config, device)
        elif isinstance(config.flip_estimator,GradientEstimatorConfig):
            self.flip_estimator = GradientFlipEstimator(config, device)
        elif isinstance(config.flip_estimator,RealFlipConfig):
            self.flip_estimator = RealFlip(config,device)

        self.device = device

    def set_device(self,
                   device):
        self.device = device
        self.flip_estimator.set_device(device)

    def __call__(self,
                 current_model: SchrodingerBridgeBackwardRate,
                 past_model: SchrodingerBridgeBackwardRate,
                 X_spins: TensorType["batch_size", "dimension"],
                 current_time: TensorType["batch_size"],
                 sinkhorn_iteration=0):
        """
        :param current_model:
        :param X_spins:
        :return:
        """

        if self.flip_old_time:
            if sinkhorn_iteration % 2 == 0:
                old_time = current_time
            else:
                old_time = 1. - current_time

        if self.flip_current_time:
            if sinkhorn_iteration % 2 == 0:
                current_time = current_time
            else:
                current_time = 1. - current_time

        phi_new_d = current_model.flip_rate(X_spins, current_time).squeeze()
        with torch.no_grad():
            phi_old_d = past_model.flip_rate(X_spins, old_time)
            phi_old_d = phi_old_d.squeeze()

        # stein estimate
        stein_estimate = self.flip_estimator(current_model, X_spins, current_time)

        # stein estimate
        loss = (phi_new_d ** 2) - (2. * stein_estimate * phi_old_d)

        if self.dimension is not None:
            loss_d = loss[:,self.dimension]
            return loss_d.mean()
        else:
            return loss.mean()


if __name__=="__main__":
    from torch.distributions import Bernoulli
    from graph_bridges.configs.config_sb import SBConfig
    from graph_bridges.models.losses.loss_configs import GradientEstimatorConfig,SteinSpinEstimatorConfig
    from graph_bridges.models.backward_rates.backward_rate_utils import load_backward_rates

    config = SBConfig
    config.data.number_of_spins = 2

    number_of_spins = 2
    batch_size = 3
    spins = Bernoulli(torch.full((number_of_spins,),.5)).sample()
