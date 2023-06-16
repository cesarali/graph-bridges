import torch
from torch.distributions import Bernoulli
from graph_bridges.utils.spin_utils import bool_to_spins
class SteinSpinEstimator:
    """
    """
    def __init__(self,stein_epsilon=1e-3,
                 stein_sample_size=1000,
                 number_of_spins=3,
                 device=torch.device("cpu"),
                 **kwargs):
        self.stein_epsilon = stein_epsilon
        self.stein_sample_size = stein_sample_size
        self.epsilon_distribution = Bernoulli(torch.full((number_of_spins,), self.stein_epsilon))
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

    def estimator_static(self, phi, X, current_time):
        """
        This function calculates $\phi(x_{\d},-x_d)$ i.e.
        for a data sample X torch.Tensor of size (number_of_paths,number_of_spins)

        we return

        Args
            X torch.Tensor(size=(number_of_paths,number_of_spins), {1.,-1.})

         :returns
            estimator torch.Tensor(size=number_of_paths*number_of_spins,1))
        """
        # HERE WE MAKE SURE THAT THE DATA IS IN SPINS
        if X.dtype == torch.bool:
            X = bool_to_spins(X)
        S = self.epsilon_distribution.sample(sample_shape=(self.stein_sample_size,)).bool().to(self.device)
        S = ~S  # Manfred's estimators requieres epsilon as the probability for -1.
        S = bool_to_spins(S)

        number_of_paths = X.shape[0]
        number_of_spins = X.shape[1]

        # ESTIMATOR
        X_stein_copy,S_stein_copy,current_time = self.stein_sample_per_sample_point(X,S,current_time)
        stein_estimator = phi(S_stein_copy * X_stein_copy,current_time)
        stein_estimator = (1. - S_stein_copy) * stein_estimator
        stein_estimator = stein_estimator.reshape(number_of_paths,
                                                  self.stein_sample_size,
                                                  number_of_spins)

        stein_estimator = stein_estimator.mean(axis=1)
        stein_estimator = (1 / (2. * self.stein_epsilon)) * stein_estimator
        stein_estimator = stein_estimator.reshape(number_of_paths * number_of_spins, 1)

        return stein_estimator


class BackwardRatioEstimator:
    name_ = "backward_stein_estimator"
    def __init__(self,**kwargs):
        stein_parameters = kwargs.get("stein_parameters")
        self.dimension = kwargs.get("dimension",1)
        self.estimator_ = kwargs.get("estimator","dynamic")
        self.stein_estimator = SteinSpinEstimator(**stein_parameters)
        self.parameters_ = kwargs
        self.parameters_.update({"name": self.name_})
        self.device = torch.device("cpu")

    def set_device(self,device):
        self.device = device
        self.stein_estimator.set_device(device)

    def obtain_parameters(self):
        return self.parameters_

    def estimator_dynamic(self, current_model, past_model, paths_batch, time_grid):
        """
        :param current_model:
        :param X_spins:
        :return:
        """
        phi_new_d = current_model(paths_batch, time_grid).squeeze()
        with torch.no_grad():
            phi_old_d = past_model(paths_batch, time_grid).squeeze()

        # stein estimate
        stein_estimate = self.stein_estimator.estimator_dynamic(current_model, paths_batch, time_grid)
        loss = (phi_new_d ** 2) - (2. * stein_estimate * phi_old_d)
        if self.dimension is not None:
            loss_d = loss[:,self.dimension]
            return loss_d.mean()
        else:
            return loss.mean()

    def estimator_static(self,model_forward,model_backward,X_spins,current_time):
        """
        :param model_forward:
        :param X_spins:
        :return:
        """
        batch_size = X_spins.shape[0]
        number_of_spins = X_spins.shape[1]

        phi_new_d = model_forward(X_spins,current_time).squeeze()
        with torch.no_grad():
            phi_old_d = model_backward(X_spins,current_time).squeeze()

        # stein estimate
        stein_estimate = self.stein_estimator.estimator_static(model_forward,X_spins,current_time)
        stein_estimate = stein_estimate.reshape(batch_size, number_of_spins)
        # stein estimate
        loss = (phi_new_d ** 2) - (2. * stein_estimate * phi_old_d)

        if self.dimension is not None:
            loss_d = loss[:,self.dimension]
            return loss_d.mean()
        else:
            return loss.mean()

    def estimator(self,model_forward,model_backward,paths_batch,time_grid):
        if self.estimator_ == "dynamic":
            return self.estimator_dynamic(model_forward,model_backward,paths_batch,time_grid)
        elif self.estimator_ == "static":
            return self.estimator_static(model_forward,model_backward,paths_batch,time_grid)

    def get_parameters(self):
        parameters = {
            "stein_parameters": None,
            "dimension": None,
            "estimator": "dynamic",
            "stein_estimator": None,
            "parameters_": None,
            "name": self.name_,
            "device": torch.device("cpu")
        }
        return parameters