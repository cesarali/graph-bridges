from dataclasses import dataclass,asdict,field
from typing import List, Union, Optional, Tuple
import shutil
import time
import os
import subprocess
import json
import torch
from pathlib import Path

from graph_bridges.models.backward_rates.sb_backward_rate_config import SchrodingerBridgeBackwardRateConfig
from graph_bridges.data.graph_dataloaders_config import TargetConfig, CommunityConfig, GraphDataConfig
from graph_bridges.data.graph_dataloaders_config import all_dataloaders_configs
from graph_bridges.models.reference_process.reference_process_config import GaussianTargetRateConfig
from graph_bridges.models.reference_process.reference_process_config import all_reference_process_configs
from graph_bridges.configs.config_files import ExperimentFiles
from pprint import pprint
from graph_bridges.models.temporal_networks.temporal_networks_configs import all_temp_nets_configs
from graph_bridges.models.temporal_networks.transformers.temporal_hollow_transformers import TemporalHollowTransformerConfig
from graph_bridges.models.temporal_networks.unets.unet_wrapper import UnetTauConfig
from graph_bridges.models.temporal_networks.mlp.temporal_mlp import TemporalMLPConfig,DeepTemporalMLPConfig
from graph_bridges.data.graph_dataloaders_config import TargetConfig
from graph_bridges.models.losses.loss_configs import GradientEstimatorConfig,SteinSpinEstimatorConfig
from graph_bridges.models.losses.loss_configs import all_loss_configs
def get_git_revisions_hash():
    hashes = []
    hashes.append(subprocess.check_output(['git', 'rev-parse', 'HEAD']))
    hashes.append(subprocess.check_output(['git', 'rev-parse', 'HEAD^']))
    return hashes

#logs_dir = Path(save_dir).joinpath('tensorboard')
#logs_dir.mkdir(exist_ok=True)

@dataclass
class SBExperimentsFiles(ExperimentFiles):
    best_model_path_checkpoint:str = None
    best_model_path:str = None
    plot_path:str = None
    graph_plot_path:str = None

    def __post_init__(self):
        super().__post_init__()
        self.best_model_path_checkpoint = os.path.join(self.results_dir, "model_checkpoint_{0}_sinkhorn_{1}.tr")
        self.best_model_path = os.path.join(self.results_dir, "best_model_sinkhorn_{0}.tr")
        self.plot_path = os.path.join(self.results_dir, "path_marginal_at_site_{0}.png")
        self.graph_plot_path = os.path.join(self.results_dir, "graph_plots_{0}.png")

@dataclass
class ParametrizedSamplerConfig:
    """
    Sampler for Parametrized Rates
    """
    name:str = 'TauLeaping' # TauLeaping or PCTauLeaping
    type:str = 'doucet'
    step_type:str = 'TauLeaping' # TauLeaping, poisson
    num_steps:int = 20
    min_t:float = 0.01
    sample_from_reference_native:bool = False
    eps_ratio:float = 1e-9
    initial_dist:str = 'gaussian'
    num_corrector_steps:int = 10
    corrector_step_size_multiplier:float = 1.5
    corrector_entry_time:float = 0.1

    def define_min_t_from_number_of_steps(self):
        self.min_t = 1./self.num_steps

@dataclass
class BackwardRatioFlipEstimatorConfig:
    name : str = "BackwardRatioFlipEstimator"
    dimension_to_check : int = None

@dataclass
class SBSchedulerConfig:
    name :str = 'SBScheduler'

@dataclass
class SBPipelineConfig:
    name : str = 'SBPipeline'

@dataclass
class SBTrainerConfig:
    device:str = "cuda:0"
    number_of_paths : int = 10
    number_of_sinkhorn : int = 1
    starting_sinkhorn: int = 0

    optimizer_name :str = 'AdamW'
    max_n_iters :int = 10000
    clip_grad :bool= True
    clip_max_norm : float = 10.
    warmup :int = 50
    num_epochs :int = 50
    learning_rate :float = 2e-4

    gradient_accumulation_steps :int = 1
    lr_warmup_steps :int = 500
    save_image_epochs :int = 10
    save_model_global_iter :int = 1000
    save_metric_epochs: int = 25
    save_model_epochs :int = 25

    metrics: List[str] = field(default_factory=lambda: ["graphs", "graphs_plots", "histograms","mse_histograms"])
    exact_backward:bool=True
    histograms_on_train:bool = True
    #metrics = ["graphs","histograms"]

@dataclass
class SBConfig:
    from graph_bridges import results_path

    config_path : str = ""
    # files, directories and naming ---------------------------------------------
    delete :bool = True
    experiment_name :str = 'graph'
    experiment_type :str = 'sb'
    experiment_indentifier :str = None
    init_model_path = None

    # different elements configurations------------------------------------------
    model : SchrodingerBridgeBackwardRateConfig = SchrodingerBridgeBackwardRateConfig()
    temp_network : Union[UnetTauConfig,TemporalHollowTransformerConfig,TemporalMLPConfig,DeepTemporalMLPConfig] = TemporalMLPConfig()

    data : GraphDataConfig =  CommunityConfig() # corresponds to the distributions at start time
    target : TargetConfig =  TargetConfig() # corresponds to the distribution at final time

    reference : GaussianTargetRateConfig = GaussianTargetRateConfig()
    sampler : ParametrizedSamplerConfig = ParametrizedSamplerConfig()

    flip_estimator : Union[GradientEstimatorConfig,SteinSpinEstimatorConfig]= SteinSpinEstimatorConfig()
    loss : BackwardRatioFlipEstimatorConfig = BackwardRatioFlipEstimatorConfig()

    scheduler : SBSchedulerConfig =  SBSchedulerConfig()
    pipeline : SBPipelineConfig =  SBPipelineConfig()
    trainer : SBTrainerConfig =  SBTrainerConfig()
    experiment_files: SBExperimentsFiles = None

    number_of_paths : int = 10
    number_of_sinkhorn : int = 1

    # devices and parallelization ----------------------------------------------
    distributed = False
    num_gpus = 0

    def __post_init__(self):
        self.experiment_files = SBExperimentsFiles(delete=self.delete,
                                                   experiment_name=self.experiment_name,
                                                   experiment_indentifier=self.experiment_indentifier,
                                                   experiment_type=self.experiment_type)
        if isinstance(self.model,dict):
            self.model = SchrodingerBridgeBackwardRateConfig(**self.model)
        if isinstance(self.temp_network,dict):
            self.temp_network = all_temp_nets_configs[self.temp_network["temp_name"]](**self.temp_network)
        if isinstance(self.data,dict):
            if self.data["data"] in all_dataloaders_configs.keys():
                self.data = all_dataloaders_configs[self.data["data"]](**self.data)
            elif self.data["name"] in all_dataloaders_configs.keys():
                self.data = all_dataloaders_configs[self.data["name"]](**self.data)
        if isinstance(self.target,dict):
            if self.target["data"] in all_dataloaders_configs.keys():
                self.target = all_dataloaders_configs[self.target["data"]](**self.target)
            elif self.target["name"] in all_dataloaders_configs.keys():
                self.target = all_dataloaders_configs[self.target["name"]](**self.target)
        if isinstance(self.reference,dict):
            reference_name = self.reference["name"]
            self.reference = all_reference_process_configs[reference_name](**self.reference)
        if isinstance(self.sampler,dict):
            self.sampler = ParametrizedSamplerConfig(**self.sampler)
        if isinstance(self.scheduler,dict):
            self.scheduler = SBSchedulerConfig(**self.scheduler)
        if isinstance(self.pipeline,dict):
            self.pipeline = SBPipelineConfig(**self.pipeline)
        if isinstance(self.trainer, dict):
            self.trainer = SBTrainerConfig(**self.trainer)
        if isinstance(self.loss,dict):
            self.loss = BackwardRatioFlipEstimatorConfig(**self.loss)
        if isinstance(self.flip_estimator, dict):
            self.flip_estimator = all_loss_configs[self.flip_estimator["name"]](**self.flip_estimator)

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
        self.experiment_indentifier = self.experiment_files.experiment_indentifier
        self.config_path = self.experiment_files.config_path
        self.save_config()

    def align_configurations(self):
        """
        Here we make the transformation necesary for the data to fix the
        temporal network

        :return:
        """
        #dataloaders for training
        self.data.as_image = False
        self.data.as_spins = True
        self.target.as_spins = True

        # data distributions matches at the end
        self.target.batch_size = self.data.batch_size

        # target
        if isinstance(self.target,TargetConfig):
            self.target.S = self.data.S
            self.target.D = self.data.D
            self.target.C = self.data.C
            self.target.H = self.data.H
            self.target.W = self.data.W

    def save_config(self):
        config_as_dict = asdict(self)
        config_as_dict["experiment_files"]["results_dir"] = str(config_as_dict["experiment_files"]["results_dir"])
        config_as_dict["data"]["dir"] = str(config_as_dict["data"]["dir"])
        with open(self.experiment_files.config_path, "w") as file:
            json.dump(config_as_dict, file)

def get_sb_config_from_file(experiment_name, experiment_type, experiment_indentifier)->SBConfig:
    from graph_bridges import results_path

    experiment_dir = os.path.join(results_path, experiment_name)
    experiment_type_dir = os.path.join(experiment_dir, experiment_type)
    results_dir = os.path.join(experiment_type_dir, experiment_indentifier)
    results_dir_path = Path(results_dir)
    if results_dir_path.exists():
        config_path = os.path.join(results_dir, "config.json")
        config_path_json = json.load(open(config_path,"r"))
        config_path_json["delete"] = False
        config_path_json['experiment_indentifier'] = experiment_indentifier
        config = SBConfig(**config_path_json)
        return config
    else:
        raise Exception("Folder Does Not Exist")


if __name__=="__main__":
    experiment_indentifier = "tutorial_sb_trainer"
    experiment_name = "graph"
    config = get_sb_config_from_file(experiment_name=experiment_name,
                                     experiment_type="sb",
                                     experiment_indentifier=experiment_indentifier)
    pprint(config.__dict__)