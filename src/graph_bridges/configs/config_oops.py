from dataclasses import dataclass,asdict,field,fields
from graph_bridges.configs.config_files import ExperimentFiles0
from graph_bridges.data.image_dataloaders import NISTLoaderConfig
from graph_bridges.models.pipelines.pipelines_config import OopsPipelineConfig
from graph_bridges.data.register_configs import all_dataloaders_configs
from typing import List

import json

@dataclass
class RBMConfig:
    name:str = "RBM"
    n_visible: int  = 10
    n_hidden: int = 10

@dataclass
class ContrastiveDivergenceTrainerConfig:

    name:str = "ContrastiveDivergenceTrainer"
    learning_rate:float = 0.1
    number_of_epochs:int = 10
    device:str = "cuda:0"
    save_model_epochs:int = 1000
    save_metric_epochs:int=1000
    constrastive_diverge_sample_size = 10

    metrics:List[str] = field(default_factory=lambda:["kdmm","mse_histograms","graphs","graphs_plots"])

    def __post_init__(self):
        self.save_model_epochs = int(.25*self.number_of_epochs)
        self.save_metric_epochs = int(.25*self.number_of_epochs)

all_trainers_configs = {"ContrastiveDivergenceTrainer":ContrastiveDivergenceTrainerConfig}
all_model_configs = {"RBM":RBMConfig}

@dataclass
class OopsConfig:

    config_path : str = ""

    # files, directories and naming ---------------------------------------------
    delete :bool = True
    experiment_name :str = 'oops'
    experiment_type :str = 'mnist'
    experiment_indentifier :str  = None
    init_model_path = None

    # all configs ---------------------------------------------------------------
    model: RBMConfig = RBMConfig()
    data: NISTLoaderConfig = NISTLoaderConfig()
    trainer: ContrastiveDivergenceTrainerConfig = ContrastiveDivergenceTrainerConfig()
    pipeline: OopsPipelineConfig = OopsPipelineConfig()
    experiment_files:ExperimentFiles0 = None

    def __post_init__(self):
        if isinstance(self.experiment_files,dict):
            self.experiment_files = ExperimentFiles0(delete=self.delete,
                                                     experiment_dir=self.experiment_files["experiment_dir"])
        else:
            self.experiment_files = ExperimentFiles0(delete=self.delete,
                                                     experiment_name=self.experiment_name,
                                                     experiment_indentifier=self.experiment_indentifier,
                                                     experiment_type=self.experiment_type)

        if isinstance(self.model, dict):
            self.model = all_model_configs[self.model["name"]](**self.model)
        if isinstance(self.data, dict):
            self.data = all_dataloaders_configs[self.data["data"]](**self.data)
        if isinstance(self.trainer,dict):
            self.trainer = all_trainers_configs[self.trainer["name"]](**self.trainer)
        if isinstance(self.pipeline,dict):
            self.pipeline = OopsPipelineConfig(**self.pipeline)

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
        with open(self.experiment_files.config_path, "w") as file:
            json.dump(config_as_dict, file)
    def align_configurations(self):
        pass


