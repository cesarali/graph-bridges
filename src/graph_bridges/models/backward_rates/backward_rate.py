import torch
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from abc import ABC,abstractmethod
from graph_bridges.configs.graphs.graph_config_sb import SBConfig

from typing import Union, Tuple
from torchtyping import TensorType
from graph_bridges.models.temporal_networks import networks_tau
from graph_bridges.models.reference_process.ctdd_reference import GaussianTargetRate


from dataclasses import dataclass
from diffusers.utils import BaseOutput
from graph_bridges.configs.images.cifar10_config_ctdd import CTDDConfig

@dataclass
class BackwardRateOutput(BaseOutput):
    """
    :param BaseOutput:
    :return:
    """
    x_logits: torch.Tensor
    p0t_reg: torch.Tensor
    p0t_sig: torch.Tensor
    reg_x: torch.Tensor

class BackwardRate(nn.Module,ABC):

    def __init__(self,
                 config:CTDDConfig,
                 device,
                 rank,
                 **kwargs):
        super().__init__()

        self.config = config

        # DATA
        self.dimension = config.data.D
        self.num_states = config.data.S
        #self.data_type = config.data.type
        self.data_min_max = config.data.data_min_max

        # TIME
        self.time_embed_dim = config.temp_network.time_embed_dim
        self.act = nn.functional.silu

    def init_parameters(self):
        return None

    @abstractmethod
    def _forward(self,
                x: TensorType["batch_size", "dimension"],
                times:TensorType["batch_size"]
                )-> TensorType["batch_size", "dimension", "num_states"]:
        return None

    def _center_data(self, x):
        out = (x - self.data_min_max[0]) / (self.data_min_max[1] - self.data_min_max[0])  # [0, 1]
        return 2 * out - 1  # to put it in [-1, 1]

    def forward(self,
                x: TensorType["batch_size", "dimension"],
                times:TensorType["batch_size"],
                x_tilde: TensorType["batch_size", "dimension"] = None,
                return_dict: bool = False,
                )-> Union[BackwardRateOutput, torch.FloatTensor, Tuple]:
        if x_tilde is not None:
            return self.ctdd(x,x_tilde,times,return_dict)
        else:
            x_logits = self._forward(x, times)
            if not return_dict:
                return x_logits
            else:
                return BackwardRateOutput(x_logits=x_logits, p0t_reg=None, p0t_sig=None)

    def ctdd(self,x_t,x_tilde,times,return_dict):
        if x_tilde is not None:
            if self.config.loss.one_forward_pass:
                reg_x = x_tilde
                x_logits = self._forward(reg_x, times)  # (B, D, S)
                p0t_reg = F.softmax(x_logits, dim=2)  # (B, D, S)
                p0t_sig = p0t_reg
            else:
                reg_x = x_t
                x_logits = self._forward(reg_x, times)  # (B, D, S)
                p0t_reg = F.softmax(x_logits, dim=2)  # (B, D, S)
                p0t_sig = F.softmax(self._forward(x_tilde, times), dim=2)  # (B, D, S)

            if not return_dict:
                return (x_logits,p0t_reg,p0t_sig,reg_x)
            return BackwardRateOutput(x_logits=x_logits,p0t_reg=p0t_reg,p0t_sig=p0t_sig,reg_x=reg_x)
        else:
            return self._forward(x_t, times)  # (B, D, S)

    def flip_rate(self,
                x: TensorType["batch_size", "dimension"],
                times: TensorType["batch_size"]
                ) -> TensorType["batch_size", "dimension"]:
        forward_logits = self.forward(x,times)
        return forward_logits[:,:,0]

class BackwardRateOneBack(BackwardRate,GaussianTargetRate):
    """
    This estimator is expected to work for one sinkhorn iteration
    i.e. a simple diffussion model were we are able to sample from
    the target distribution

    """
    name_ = "one_back_per_time"

    def __init__(self,**kwargs):
        super(BackwardRateOneBack,self).__init__(**kwargs)

        #=====================================
        # TRANSFORMATIONS
        #=====================================
        self.number_of_spins = kwargs.get("number_of_spins")
        self.hidden_1 = kwargs.get("hidden_1")
        self.time_embedding_dim = kwargs.get("time_embedding_dim",9)

        self.f1 = nn.Linear(self.number_of_spins,self.hidden_1)
        self.f2 = nn.Linear(self.hidden_1+self.time_embedding_dim,self.number_of_spins)

        #=====================================
        # PROCESS
        #=====================================
        self.reference_parameters = kwargs.get("reference_process_parameters")
        self.reference_process_name = self.reference_parameters.get("reference_process_name")

        reference_process_factory = ReferenceProcessFactory()
        self.reference_process = reference_process_factory.create(self.reference_process_name,
                                                                   **self.reference_parameters)

    def parameters_of_linear_function(self,times):
        integral_ = self.reference_process.integral_rate(times)
        u_t = torch.exp(-2.*integral_)
        A = (1./(1.+u_t)) + (1./(1.-u_t))
        B = (1./(1.+u_t)) - (1./(1.-u_t))
        return A, B

    def forward_states_and_times(self, states, times):
        """

        :param states:
        :param times:
        :return:
        """
        A, B = self.parameters_of_linear_function(times)
        X_0_hat = self.one_back_regression(states,times)
        ratio = A + states*B*X_0_hat - 1.
        return softplus(ratio)

    def one_back_regression(self,states,times):
        time_embbedings = get_timestep_embedding(times.squeeze(),
                                                 time_embedding_dim=self.time_embedding_dim)
        step_one = self.f1(states)
        step_two = torch.concat([step_one, time_embbedings], dim=1)
        ratio_estimator = self.f2(step_two)
        return ratio_estimator

    def init_weights(self):
        nn.init.xavier_uniform_(self.f1.weight)
        nn.init.xavier_uniform_(self.f2.weight)

    @classmethod
    def get_parameters(self) -> dict:
        Q_sigma = 512.
        kwargs = super().get_parameters()
        kwargs.update({"hidden_1":14})
        reference_process_parameters = {
            "reference_process_name": "efficient_diffusion",
            "target": "gaussian",
            "rate_sigma": 6.0,
            "S": 2,
            "Q_sigma": Q_sigma,
            "time_exponential": 1.5,
            "time_base": 1.0,
            "device": torch.device("cpu")
        }
        kwargs.update({"reference_process_parameters":reference_process_parameters})
        return kwargs
