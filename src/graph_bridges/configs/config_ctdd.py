from dataclasses import dataclass,asdict,field
from typing import List, Union, Optional, Tuple
import shutil
import time
import json
import os
import torch

from graph_bridges.models.losses.loss_configs import CTDDLossConfig
from graph_bridges.data.graph_dataloaders_config import TargetConfig, CommunityConfig, GraphDataConfig
from graph_bridges.data.register_configs import all_dataloaders_configs
from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig
from graph_bridges.models.backward_rates.ctdd_backward_rate_config import all_backward_rates_configs
from graph_bridges.models.reference_process.reference_process_config import GaussianTargetRateConfig
from graph_bridges.models.reference_process.reference_process_config import all_reference_process_configs
from graph_bridges.configs.config_files import ExperimentFiles
from graph_bridges.models.temporal_networks.transformers.temporal_hollow_transformers import TemporalHollowTransformerConfig
from graph_bridges.models.temporal_networks.convnets.autoencoder import ConvNetAutoencoderConfig

from graph_bridges.models.temporal_networks.unets.unet_wrapper import UnetTauConfig
from graph_bridges.models.temporal_networks.temporal_networks_configs import all_temp_nets_configs

from pprint import pprint


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
class BackwardEstimatorConfig:

    name : str = "BackwardRatioSteinEstimator"
    dimension_to_check : int = None

@dataclass
class CTDDSchedulerConfig:
    name :str = 'CTDDScheduler'

@dataclass
class CTDDPipelineConfig:
    name : str = 'CTDDPipeline'

@dataclass
class CTDDTrainerConfig:
    device: str = "cuda:0"
    number_of_paths : int = 10
    num_epochs :int = 5
    distributed: bool = False

    optimizer_name :str = 'AdamW'
    max_n_iters :int = 10000
    clip_grad :bool= True
    warmup :int = 50
    learning_rate :float = 2e-4

    gradient_accumulation_steps :int = 1
    lr_warmup_steps :int = 500
    save_metric_epochs: int = None
    save_image_epochs :int = None
    save_model_epochs :int = None
    save_model_global_iter :int = None
    log_loss: int = 500

    metrics:List[str] = field(default_factory=lambda: ["graphs", "graphs_plots", "histograms","mse_histograms"])

    def __post_init__(self):
        self.save_metric_epochs = max(int(.25*self.num_epochs),1)
        self.save_image_epochs = max(int(.25*self.num_epochs),1)
        self.save_model_epochs = max(int(.25*self.num_epochs),1)
        self.save_model_global_iter = max(int(.25*self.num_epochs),1)

@dataclass
class CTDDExperimentsFiles(ExperimentFiles):
    best_model_path_checkpoint:str = None
    best_model_path:str = None
    plot_path:str = None
    graph_plot_path:str = None

    def __post_init__(self):
        super().__post_init__()

        self.best_model_path_checkpoint = os.path.join(self.results_dir, "model_checkpoint_{0}.tr")
        self.best_model_path = os.path.join(self.results_dir, "best_model.tr")
        self.plot_path = os.path.join(self.results_dir, "marginal_at_site_{0}.png")
        self.graph_plot_path = os.path.join(self.results_dir, "graph_plots_{0}.png")


@dataclass
class CTDDConfig:

    config_path : str = ""
    # different elements configurations------------------------------------------
    model : BackRateMLPConfig = BackRateMLPConfig()
    temp_network : Union[UnetTauConfig,TemporalHollowTransformerConfig,ConvNetAutoencoderConfig] = UnetTauConfig()

    data : GraphDataConfig = CommunityConfig() # corresponds to the distributions at start time
    target : TargetConfig = TargetConfig() # corresponds to the distribution at final time

    reference : GaussianTargetRateConfig =  GaussianTargetRateConfig()
    sampler : ParametrizedSamplerConfig =  ParametrizedSamplerConfig()

    loss : CTDDLossConfig =  CTDDLossConfig()
    scheduler : CTDDSchedulerConfig = CTDDSchedulerConfig()
    pipeline : CTDDPipelineConfig = CTDDPipelineConfig()
    trainer : CTDDTrainerConfig = CTDDTrainerConfig()
    experiment_files: CTDDExperimentsFiles = None

    number_of_paths : int = 10

    delete :bool = False
    experiment_type: str = 'ctdd'
    experiment_name :str = 'general'
    experiment_indentifier :str  = 'testing'
    init_model_path = None

    def __post_init__(self):
        if isinstance(self.experiment_files,dict):
            self.experiment_files = CTDDExperimentsFiles(delete=self.delete,
                                                       results_dir=self.experiment_files["results_dir"])
        else:
            self.experiment_files = CTDDExperimentsFiles(delete=self.delete,
                                                       experiment_name=self.experiment_name,
                                                       experiment_indentifier=self.experiment_indentifier,
                                                       experiment_type=self.experiment_type)
        #self.experiment_files.data_stats = os.path.join(self.data.preprocess_datapath, "data_stats.json")

        if isinstance(self.model,dict):
            self.model = all_backward_rates_configs[self.model["name"]](**self.model)
        if isinstance(self.temp_network,dict):
            self.temp_network = all_temp_nets_configs[self.temp_network["temp_name"]](**self.temp_network)
        if isinstance(self.data,dict):
            self.data =  all_dataloaders_configs[self.data["data"]](**self.data)
        if isinstance(self.target,dict):
            self.target = all_dataloaders_configs[self.target["data"]](**self.target)  # corresponds to the distribution at final time
        if isinstance(self.reference,dict):
            self.reference = all_reference_process_configs[self.reference["name"]](**self.reference)
        if isinstance(self.sampler,dict):
            self.sampler = ParametrizedSamplerConfig(**self.sampler)
        if isinstance(self.loss,dict):
            self.loss = CTDDLossConfig(**self.loss)
        if isinstance(self.scheduler,dict):
            self.scheduler = CTDDSchedulerConfig(**self.scheduler)
        if isinstance(self.pipeline,dict):
            self.pipeline = CTDDPipelineConfig(**self.pipeline)
        if isinstance(self.trainer, dict):
            self.trainer = CTDDTrainerConfig(**self.trainer)

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

    def save_config(self):
        config_as_dict = asdict(self)
        config_as_dict["experiment_files"]["results_dir"] = str(config_as_dict["experiment_files"]["results_dir"])
        try:
            config_as_dict["data"]["dir"] = str(config_as_dict["data"]["dir"])
        except:
            pass

        with open(self.experiment_files.config_path, "w") as file:
            json.dump(config_as_dict, file)

    #def create_directories(self):
    #    self.experiment_files = ExperimentFiles(experiment_indentifier=self.experiment_indentifier,
    #                                            experiment_name=self.experiment_name,
    #                                            experiment_type=self.experiment_type,
    #                                            delete=self.delete)
    #    self.results_dir = self.experiment_files.results_dir

    def align_configurations(self):
        from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig, BackwardRateTemporalHollowTransformerConfig
        from graph_bridges.models.temporal_networks.transformers.temporal_hollow_transformers import TemporalHollowTransformerConfig
        from graph_bridges.models.backward_rates.ctdd_backward_rate_config import GaussianTargetRateImageX0PredEMAConfig

        from graph_bridges.models.temporal_networks.convnets.autoencoder import ConvNetAutoencoderConfig
        from graph_bridges.models.temporal_networks.mlp.temporal_mlp import TemporalMLPConfig
        from graph_bridges.models.temporal_networks.unets.unet_wrapper import UnetTauConfig

        if isinstance(self.model,BackRateMLPConfig):
            self.data.as_image = False
            self.data.as_spins = False
            if not isinstance(self.temp_network,TemporalMLPConfig):
                self.temp_network = TemporalMLPConfig()

        elif isinstance(self.model,GaussianTargetRateImageX0PredEMAConfig):
            if isinstance(self.temp_network,ConvNetAutoencoderConfig):
                #dataloaders for training
                self.data.as_spins = False
                self.data.as_image = True
                self.data.flatten_adjacency = False
            elif isinstance(self.temp_network, UnetTauConfig):
                raise Exception("Unet Network not implemented for Graphs (Yet)")

        elif isinstance(self.model, BackwardRateTemporalHollowTransformerConfig):
            self.data.as_spins = False
            self.data.as_image = False
            self.data.flatten_adjacency = True
            if not isinstance(self.temp_network,TemporalHollowTransformerConfig):
                self.temp_network = TemporalHollowTransformerConfig(input_vocab_size=2,
                                                                    output_vocab_size=2,
                                                                    max_seq_length=self.data.D)
            else:
                self.temp_network : TemporalHollowTransformerConfig
                self.temp_network.input_vocab_size = 2
                self.temp_network.output_vocab_size = 2
                self.temp_network.max_seq_length = self.data.D

        self.data.__post_init__()
        # data distributions matches at the end
        self.target.batch_size = self.data.batch_size

        # model matches reference process
        self.reference.initial_dist = self.model.initial_dist
        self.reference.rate_sigma = self.model.rate_sigma
        self.reference.Q_sigma = self.model.Q_sigma
        self.reference.time_exponential = self.model.time_exponential
        self.reference.time_base = self.model.time_base

def get_config_from_file(experiment_name,experiment_type,experiment_indentifier)->CTDDConfig:
    from graph_bridges import results_path

    experiment_dir = os.path.join(results_path, experiment_name)
    experiment_type_dir = os.path.join(experiment_dir, experiment_type)
    results_dir = os.path.join(experiment_type_dir, experiment_indentifier)

    config_path = os.path.join(results_dir, "config.json")
    config_path_json = json.load(open(config_path,"r"))
    config_path_json["delete"] = False

    config = CTDDConfig(**config_path_json)

    return config
