import os
import torch
from pathlib import Path
from dataclasses import dataclass

@dataclass
class GaussianTargetRateConfig:
    """
    Reference configuration for schrodinger bridge reference process
    """
    # reference process variables
    name:str = "GaussianTargetRate"
    initial_dist:str = 'gaussian'
    rate_sigma:float = 6.0
    Q_sigma:float = 512.0
    time_exponential:float = 3.
    time_base:float = 1.0

    def __init__(self,
                 name="GaussianTargetRate",
                 initial_dist='gaussian',
                 rate_sigma=6.0,
                 Q_sigma=512.0,
                 time_exponential=3.0,
                 time_base=1.0,
                 **kwargs):

        self.name = name
        self.initial_dist = initial_dist
        self.rate_sigma = rate_sigma
        self.Q_sigma = Q_sigma
        self.time_exponential = time_exponential
        self.time_base = time_base


all_reference_process_configs = {"GaussianTargetRate":GaussianTargetRateConfig}