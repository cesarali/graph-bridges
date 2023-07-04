from graph_bridges.configs.graphs.lobster.config_base import BridgeConfig
from graph_bridges.models.reference_process.reference_process_utils import register_reference
import torch
import math

from typing import Tuple,Union
from torchtyping import TensorType
import numpy as np

class ReferenceProcess:
    """
    """
    def __init__(self,config:BridgeConfig,device):
        self.S = config.data.S
        self.D = config.data.D
        self.eps_ratio = config.sampler.eps_ratio
        self.device = device

    def rates(self,
              x:TensorType["num_of_paths", "dimensions"],
              t:TensorType["num_of_paths"],
              device=None) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        """

        :param x:
        :param t:
        :return: forward_rates,qt0_denom,qt0_numer
        """
        if device is None:
            device = self.device

        num_of_paths = x.shape[0]
        qt0 = self.transition(t * torch.ones((num_of_paths,), device=device))  # (N, S, S)
        rate = self.rate(t * torch.ones((num_of_paths,), device=device))  # (N, S, S)

        forward_rates = rate[
            torch.arange(num_of_paths, device=device).repeat_interleave(self.D * self.S),
            torch.arange(self.S, device=device).repeat(num_of_paths * self.D),
            x.long().flatten().repeat_interleave(self.S)
        ].view(num_of_paths, self.D, self.S)

        qt0_denom = qt0[
                        torch.arange(num_of_paths, device=device).repeat_interleave(self.D * self.S),
                        torch.arange(self.S, device=device).repeat(num_of_paths * self.D),
                        x.long().flatten().repeat_interleave(self.S)
                    ].view(num_of_paths, self.D, self.S) + self.eps_ratio

        # First S is x0 second S is x tilde
        qt0_numer = qt0  # (N, S, S)

        return forward_rates,qt0_denom,qt0_numer

    def backward_rates(self,p0t,x,t,device):
        num_of_paths = x.shape[0]
        forward_rates,qt0_denom,qt0_numer = self.rates(x,t,device)
        inner_sum = (p0t / qt0_denom) @ qt0_numer  # (N, D, S)
        backward_rates = forward_rates * inner_sum  # (N, D, S)
        backward_rates[
            torch.arange(num_of_paths, device=device).repeat_interleave(self.D),
            torch.arange(self.D, device=device).repeat(num_of_paths),
            x.long().flatten()
        ] = 0.0
        return backward_rates

    def _integral_rate_scalar(self, t: TensorType["B"]
                              ) -> TensorType["B"]:
        return None

    def _rate_scalar(self,
                     t: TensorType["B"]
                     ) -> TensorType["B"]:
        return None

    def rate(self,
             t: TensorType["B"]
             ) -> TensorType["B", "S", "S"]:
        return None

    def transition(self,
                   t: TensorType["B"]
                   ) -> TensorType["B", "S", "S"]:
        return None

@register_reference
class GaussianTargetRate(ReferenceProcess):
    def __init__(self, cfg, device,rank=None):
        ReferenceProcess.__init__(self,cfg,device)
        self.S = S = cfg.data.S
        self.rate_sigma = cfg.model.rate_sigma
        self.Q_sigma = cfg.model.Q_sigma
        self.time_exponential = cfg.model.time_exponential
        self.time_base = cfg.model.time_base
        self.device = device

        rate = np.zeros((S, S))

        vals = np.exp(-np.arange(0, S) ** 2 / (self.rate_sigma ** 2))
        for i in range(S):
            for j in range(S):
                if i < S // 2:
                    if j > i and j < S - i:
                        rate[i, j] = vals[j - i - 1]
                elif i > S // 2:
                    if j < i and j > -i + S - 1:
                        rate[i, j] = vals[i - j - 1]
        for i in range(S):
            for j in range(S):
                if rate[j, i] > 0.0:
                    rate[i, j] = rate[j, i] * np.exp(
                        - ((j + 1) ** 2 - (i + 1) ** 2 + S * (i + 1) - S * (j + 1)) / (2 * self.Q_sigma ** 2))

        rate = rate - np.diag(np.diag(rate))
        rate = rate - np.diag(np.sum(rate, axis=1))

        eigvals, eigvecs = np.linalg.eig(rate)
        inv_eigvecs = np.linalg.inv(eigvecs)

        self.base_rate = torch.from_numpy(rate).float().to(self.device)
        self.eigvals = torch.from_numpy(eigvals).float().to(self.device)
        self.eigvecs = torch.from_numpy(eigvecs).float().to(self.device)
        self.inv_eigvecs = torch.from_numpy(inv_eigvecs).float().to(self.device)

    def to(self,device):
        self.device = device
        return self

    def _integral_rate_scalar(self, t: TensorType["B"]
                              ) -> TensorType["B"]:
        return self.time_base * (self.time_exponential ** t) - \
            self.time_base

    def _rate_scalar(self, t: TensorType["B"]
                     ) -> TensorType["B"]:
        return self.time_base * math.log(self.time_exponential) * \
            (self.time_exponential ** t)

    def rate(self, t: TensorType["B"]
             ) -> TensorType["B", "S", "S"]:
        B = t.shape[0]
        S = self.S
        rate_scalars = self._rate_scalar(t)

        return self.base_rate.view(1, S, S) * rate_scalars.view(B, 1, 1)

    def transition(self, t: TensorType["B"]
                   ) -> TensorType["B", "S", "S"]:
        B = t.shape[0]
        S = self.S

        integral_rate_scalars = self._integral_rate_scalar(t)

        adj_eigvals = integral_rate_scalars.view(B, 1) * self.eigvals.view(1, S)

        transitions = self.eigvecs.view(1, S, S) @ \
                      torch.diag_embed(torch.exp(adj_eigvals)) @ \
                      self.inv_eigvecs.view(1, S, S)

        # Some entries that are supposed to be very close to zero might be negative
        if torch.min(transitions) < -1e-6:
            print(f"[Warning] GaussianTargetRate, large negative transition values {torch.min(transitions)}")

        # Clamping at 1e-8 because at float level accuracy anything lower than that
        # is probably inaccurate and should be zero anyway
        transitions[transitions < 1e-8] = 0.0

        return transitions