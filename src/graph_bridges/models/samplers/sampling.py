from graph_bridges.configs.graphs.lobster.config_base import BridgeConfig
from graph_bridges.models.backward_rates.backward_rate import BackwardRate
from graph_bridges.models.schedulers.scheduling_sb import SBScheduler
from graph_bridges.data.dataloaders import BridgeDataLoader
from pprint import pprint
from graph_bridges.models.reference_process.ctdd_reference import ReferenceProcess, GaussianTargetRate

import numpy as np
import torch


from torch.distributions.poisson import Poisson
import torch.nn.functional as F

from tqdm import tqdm
from typing import Tuple
from torchtyping import TensorType

def copy_models(model_to, model_from):
    model_to.load_state_dict(model_from.state_dict())

check_model_devices = lambda x: x.parameters().__next__().device


class NaivePoisson:

    def __init__(self):
        return None

    def sample(self,
               spins_start:TensorType["batch_size","dimension","num_states"],
               backward_model:BackwardRate,
               scheduler:SBScheduler,
               forward:bool=True,
               max_iterations=1000):
        """
        :param spins_start: torch.Tensor(size=(batch_size,number_of_spins))
        :param backward_model:
        :param T:
        :param tau:
        :return: paths, time_grid

            torch.Tensor(size=(batch_size,number_of_timesteps,number_of_spins)),
            torch.Tensor(size=(number_of_timesteps))

        """
        assert len(spins_start.shape) == 2
        device = spins_start.device

        # NAIVE POISSON
        time_grid = torch.arange(0., self.T + self.tau, self.tau).to(device)
        number_of_time_steps = len(time_grid) - 1

        # Initialize process and time management --------------------------
        paths = torch.clone(spins_start)
        paths = paths.unsqueeze(1)

        current_time_index,time_sign,end_condition = scheduler.time_direction(forward)

        # Simulation--------------------------------------------------------
        number_of_iterations = 0
        while end_condition(current_time_index):
            # evaluates times where you are heading
            current_time = time_grid[current_time_index + time_sign]
            current_ratios = backward_model(paths[:, -1, :], current_time)

            try:
                poisson_probabilities = current_ratios * self.tau
                events = Poisson(poisson_probabilities).sample()
                where_to_flip = torch.where(events > 0)
            except:
                print(number_of_iterations)

            # Copy last state
            paths = torch.concatenate([paths,
                                       paths[:, -1, :].unsqueeze(1)], dim=1)
            # Flip accordingly
            paths[:, -1, :][where_to_flip] = paths[:, -1, :][where_to_flip] * (-1.)
            current_time_index = time_sign + current_time_index

            number_of_iterations += 1
            if number_of_iterations > max_iterations:
                print("Next Reaction Time Stop Before Reaching End at ALl Paths")
                break

        if forward:
            return paths, time_grid
        else:
            paths = torch.flip(paths, dims=[1])
            return paths, time_grid

class TauLeaping:
    """
    """
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.C,self.H,self.W = self.config.data.shape
        self.D = self.config.data.D
        self.S = self.config.data.S
        self.sample_config = self.config.sampler
        self.num_steps = self.sample_config.num_steps
        self.min_t = self.sample_config.min_t

    def sample(self,
               model,
               reference_process: ReferenceProcess,
               data: BridgeDataLoader,
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
                forward_rates,qt0_denom,qt0_numer = reference_process.foward_rates_and_probabilities(x, t)
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

class DiffusersBackwardProcess:
    def __init__(self):
        return None

class DoucetForward:
    def __init__(self):
        return None


if __name__=="__main__":
    from graph_bridges.models.backward_rates.backward_rate import GaussianTargetRateImageX0PredEMA
    from graph_bridges.data.dataloaders_utils import create_dataloader

    config = BridgeConfig()
    device = torch.device(config.device)

    data_dataloader = create_dataloader(config,device)
    model = GaussianTargetRateImageX0PredEMA(config,device)
    reference_process = GaussianTargetRate(config,device)

    sampler = TauLeaping(config)
    path = sampler.sample(model,reference_process,data_dataloader,10,2)
    print(path)



