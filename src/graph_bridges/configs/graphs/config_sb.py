from dataclasses import dataclass,asdict,field
from typing import List, Union, Optional, Tuple
import shutil
import time
import os
import subprocess
import json
import torch

from graph_bridges.data.graph_dataloaders_config import TargetConfig, CommunityConfig, GraphDataConfig
from graph_bridges.data.graph_dataloaders_config import all_dataloaders_configs
from graph_bridges.models.backward_rates.backward_rate_config import BackRateMLPConfig
from graph_bridges.models.backward_rates.backward_rate_config import all_backward_rates_configs
from graph_bridges.models.reference_process.reference_process_config import GaussianTargetRateConfig
from graph_bridges.models.reference_process.reference_process_config import all_reference_process_configs
from graph_bridges.configs.files_config import ExperimentFiles
from pprint import pprint


def get_git_revisions_hash():
    hashes = []
    hashes.append(subprocess.check_output(['git', 'rev-parse', 'HEAD']))
    hashes.append(subprocess.check_output(['git', 'rev-parse', 'HEAD^']))
    return hashes

#logs_dir = Path(save_dir).joinpath('tensorboard')
#logs_dir.mkdir(exist_ok=True)

@dataclass
class SBExperimentsFiles(ExperimentFiles):

    def __post_init__(self):
        super().__post_init__()
        self.best_model_path_checkpoint = os.path.join(self.results_dir, "model_checkpoint_{0}_sinkhorn_{1}.tr")
        self.best_model_path = os.path.join(self.results_dir, "best_model_sinkhorn_{0}.tr")
        self.plot_path = os.path.join(self.results_dir, "marginal_at_site_{0}.png")
        self.graph_plot_path = os.path.join(self.results_dir, "graph_plots_{0}.png")

@dataclass
class ParametrizedSamplerConfig:
    """
    Sampler for Parametrized Rates
    """
    name:str = 'TauLeaping' # TauLeaping or PCTauLeaping
    type:str = 'doucet'
    num_steps:int = 20
    min_t:float = 0.01
    eps_ratio:float = 1e-9
    initial_dist:str = 'gaussian'
    num_corrector_steps:int = 10
    corrector_step_size_multiplier:float = 1.5
    corrector_entry_time:float = 0.1

@dataclass
class SteinSpinEstimatorConfig:

    name : str = "SteinSpinEstimator"
    stein_epsilon :float = 1e-3
    stein_sample_size :int = 150

@dataclass
class BackwardEstimatorConfig:

    name : str = "BackwardRatioSteinEstimator"
    dimension_to_check : int = None

@dataclass
class SBSchedulerConfig:
    name :str = 'SBScheduler'

@dataclass
class SBPipelineConfig:
    name : str = 'SBPipeline'

@dataclass
class TrainerConfig:
    number_of_paths : int = 10
    number_of_sinkhorn : int = 1
    starting_sinkhorn: int = 0

    optimizer_name :str = 'AdamW'
    max_n_iters :int = 10000
    clip_grad :bool= True
    warmup :int = 50
    num_epochs :int = 200
    learning_rate :float = 2e-4

    gradient_accumulation_steps :int = 1
    lr_warmup_steps :int = 500
    save_image_epochs :int = 10
    save_model_epochs :int = 30
    save_model_global_iter :int = 1000

@dataclass
class SBConfig:

    from graph_bridges import results_path

    config_path : str = ""
    # files, directories and naming ---------------------------------------------
    delete :bool = False
    experiment_name :str = 'graph'
    experiment_type :str = 'sb'
    experiment_indentifier :str  = 'testing'
    init_model_path = None

    # different elements configurations------------------------------------------
    model : BackRateMLPConfig =  BackRateMLPConfig()
    data : GraphDataConfig =  CommunityConfig() # corresponds to the distributions at start time
    target : TargetConfig =  TargetConfig() # corresponds to the distribution at final time

    reference : GaussianTargetRateConfig =  GaussianTargetRateConfig()
    sampler : ParametrizedSamplerConfig =  ParametrizedSamplerConfig()

    stein : SteinSpinEstimatorConfig= SteinSpinEstimatorConfig()
    loss : BackwardEstimatorConfig = BackwardEstimatorConfig()

    scheduler : SBSchedulerConfig =  SBSchedulerConfig()
    pipeline : SBPipelineConfig =  SBPipelineConfig()
    optimizer : TrainerConfig =  TrainerConfig()
    experiment_files: SBExperimentsFiles = SBExperimentsFiles(delete=delete,
                                                              experiment_name=experiment_name,
                                                              experiment_indentifier=experiment_indentifier,
                                                              experiment_type=experiment_type)
    number_of_paths : int = 10
    number_of_sinkhorn : int = 1

    # devices and parallelization ----------------------------------------------
    device = 'cpu'
    device_paths = 'cpu'
    distributed = False
    num_gpus = 0

    def __post_init__(self):
        if isinstance(self.model,dict):
            self.model = all_backward_rates_configs[self.model["name"]](**self.model)
        if isinstance(self.data,dict):
            self.data =  all_dataloaders_configs[self.data["data"]](**self.data)
        if isinstance(self.target,dict):
            self.target = all_dataloaders_configs[self.target["data"]](**self.target)  # corresponds to the distribution at final time
        if isinstance(self.reference,dict):
            reference_name = self.reference["name"]
            self.reference = all_reference_process_configs[reference_name](**self.reference)
        if isinstance(self.sampler,dict):
            self.sampler = ParametrizedSamplerConfig(**self.sampler)
        if isinstance(self.scheduler,dict):
            self.scheduler = SBSchedulerConfig(**self.scheduler)
        if isinstance(self.pipeline,dict):
            self.pipeline = SBPipelineConfig(**self.pipeline)
        if isinstance(self.optimizer,dict):
            self.optimizer = TrainerConfig(**self.optimizer)

        self.experiment_files.data_stats = os.path.join(self.data.preprocess_datapath, "data_stats.json")

    def initialize_new_experiment(self,
                                  experiment_name: str = None,
                                  experiment_type: str = None,
                                  experiment_indentifier: str = None):
        if experiment_name is not None:
            self.experiment_name = experiment_name
        if experiment_type is not None:
            self.experiment_type = experiment_type
        if experiment_indentifier is not None:
            self.experiment_indentifier = experiment_indentifier

        self.align_configurations()
        self.experiment_files.create_directories()
        self.config_path = self.experiment_files.config_path
        self.save_config()

    def align_configurations(self):
        #dataloaders for training
        self.data.as_image = False
        self.data.as_spins = True

        # data distributions matches at the end
        self.target.batch_size = self.data.batch_size

        # target
        self.target.S = self.data.S
        self.target.D = self.data.D
        self.target.C = self.data.C
        self.target.H = self.data.H
        self.target.W = self.data.W

        # model matches reference process
        self.reference.initial_dist = self.model.initial_dist
        self.reference.rate_sigma = self.model.rate_sigma
        self.reference.Q_sigma = self.model.Q_sigma
        self.reference.time_exponential = self.model.time_exponential
        self.reference.time_base = self.model.time_base


    def save_config(self):
        config_as_dict = asdict(self)
        config_as_dict["experiment_files"]["results_dir"] = str(config_as_dict["experiment_files"]["results_dir"])
        config_as_dict["data"]["dir"] = str(config_as_dict["data"]["dir"])
        with open(self.experiment_files.config_path, "w") as file:
            json.dump(config_as_dict, file)


if __name__=="__main__":
    from pprint import pprint
    from graph_bridges.data.graph_dataloaders_config import EgoConfig, GraphSpinsDataLoaderConfig, TargetConfig
    from graph_bridges.models.backward_rates.backward_rate_config import BackRateMLPConfig, GaussianTargetRateImageX0PredEMAConfig

    device = torch.device("cpu")

    config = SBConfig()
    config.data = EgoConfig(as_image=False, batch_size=32, full_adjacency=False)
    config.model = GaussianTargetRateImageX0PredEMAConfig(time_embed_dim=32, fix_logistic=False)
    config.initialize_new_experiment()
