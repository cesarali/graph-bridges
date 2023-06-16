from configs.graphs.lobster.config import BridgeConfig
import numpy as np
import torch

import torch.nn.functional as F
from tqdm import tqdm
from typing import List,Tuple
from torchtyping import TensorType

class BridgeData:
    config : BridgeConfig

    def __init__(self,config:BridgeConfig):
        self.config = config
        self.device = self.config.device

        C,H,W = self.config.data.shape
        self.D = C*H*W
        self.S = self.config.data.S
        sampler_config = self.config.sampler

        self.initial_dist = sampler_config.initial_dist
        if self.initial_dist == 'gaussian':
            self.initial_dist_std  = self.config.model.Q_sigma
        else:
            self.initial_dist_std = None

    def sample(self, num_of_paths, device=None):
        if device is None:
            device = self.device

        if self.initial_dist == 'uniform':
            x = torch.randint(low=0, high=self.S, size=(num_of_paths, self.D), device=device)
        elif self.initial_dist == 'gaussian':
            target = np.exp(
                - ((np.arange(1, self.S + 1) - self.S // 2) ** 2) / (2 * self.initial_dist_std ** 2)
            )
            target = target / np.sum(target)

            cat = torch.distributions.categorical.Categorical(
                torch.from_numpy(target)
            )
            x = cat.sample((num_of_paths * self.D,)).view(num_of_paths, self.D)
            x = x.to(device)
        else:
            raise NotImplementedError('Unrecognized initial dist ' + self.initial_dist)
        return x

class ReferenceProcess:
    """
    """
    def __init__(self,model,config:BridgeConfig):
        self.S = config.data.S
        self.D = config.data.D
        self.eps_ratio = config.sampler.eps_ratio
        self.model = model

    def rates(self,
              x:TensorType["num_of_paths", "dimensions"],
              t:TensorType["num_of_paths"]) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        """

        :param x:
        :param t:
        :return: forward_rates,qt0_denom,qt0_numer
        """
        num_of_paths = x.shape[0]
        qt0 = model.transition(t * torch.ones((num_of_paths,), device=device))  # (N, S, S)
        rate = model.rate(t * torch.ones((num_of_paths,), device=device))  # (N, S, S)

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

class TauLeaping():
    """
    """
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.C,self.H,self.W = self.config.data.shape
        self.D = self.config.data.D
        self.S = self.config.data.S
        self.sample_config = self.config.sampler
        self.num_steps = self.sample_config .num_steps
        self.min_t = self.sample_config .min_t

    def sample(self,
               model,
               reference_process: ReferenceProcess,
               data: BridgeData,
               num_of_paths: int,
               num_intermediates: int):
        t = 1.0

        device = model.device
        with torch.no_grad():
            x = data.sample(num_of_paths,device)

            ts = np.concatenate((np.linspace(1.0, self.min_t, self.num_steps), np.array([0])))
            save_ts = ts[np.linspace(0, len(ts)-2, num_intermediates, dtype=int)]

            x_hist = []
            x0_hist = []

            counter = 0
            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx+1]

                forward_rates,qt0_denom,qt0_numer = reference_process.rates(x,t)
                p0t = F.softmax(model(x, t * torch.ones((num_of_paths,), device=device)), dim=2) # (N, D, S)

                x_0max = torch.max(p0t, dim=2)[1]
                if t in save_ts:
                    x_hist.append(x.clone().detach().cpu().numpy())
                    x0_hist.append(x_0max.clone().detach().cpu().numpy())

                inner_sum = (p0t / qt0_denom) @ qt0_numer # (N, D, S)
                reverse_rates = forward_rates * inner_sum # (N, D, S)

                reverse_rates[
                    torch.arange(num_of_paths, device=device).repeat_interleave(self.D),
                    torch.arange(self.D, device=device).repeat(num_of_paths),
                    x.long().flatten()
                ] = 0.0

                diffs = torch.arange(self.S, device=device).view(1,1,self.S) - x.view(num_of_paths, self.D, 1)
                poisson_dist = torch.distributions.poisson.Poisson(reverse_rates * h)
                jump_nums = poisson_dist.sample()
                adj_diffs = jump_nums * diffs
                overall_jump = torch.sum(adj_diffs, dim=2)
                xp = x + overall_jump
                x_new = torch.clamp(xp, min=0, max=self.S-1)

                x = x_new

            x_hist = np.array(x_hist).astype(int)
            x0_hist = np.array(x0_hist).astype(int)

            logits = model(x, self.min_t * torch.ones((num_of_paths,), device=device))
            p_0gt = F.softmax(logits, dim=2) # (N, D, S)
            x_0max = torch.max(p_0gt, dim=2)[1]

            return x_0max.detach().cpu().numpy().astype(int), x_hist, x0_hist


if __name__=="__main__":
    from graph_bridges.models.backward_rate import ImageX0PredBase,GaussianTargetRateImageX0PredEMA

    config = BridgeConfig()
    device = torch.device(config.device)

    data = BridgeData(config)
    model = GaussianTargetRateImageX0PredEMA(config,device)
    reference_process = ReferenceProcess(model,config)

    sampler = TauLeaping(config)
    path = sampler.sample(model,reference_process,data,10,2)
    print(path)



