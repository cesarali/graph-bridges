import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict

from graph_bridges.configs.config_files import ExperimentFiles
from graph_bridges.models.networks_arquitectures.encoder_config import EncoderConfig
from graph_bridges.models.networks_arquitectures.decoder_config import DecoderConfig
from graph_bridges.trainers.vae_trainer_config import VAETrainerConfig
from graph_bridges.data.graph_dataloaders_config import all_dataloaders_configs


all_encoders_configs = {"Encoder":EncoderConfig}
all_decoders_configs = {"Decoder":DecoderConfig}
all_trainers_configs = {"VAETrainer":VAETrainerConfig}


@dataclass
class VAE_ExperimentsFiles(ExperimentFiles):
    best_model_path_checkpoint:str = None
    best_model_path:str = None
    plot_path:str = None

    def __post_init__(self):
        super().__post_init__()
        self.best_model_path_checkpoint = os.path.join(self.results_dir, "model_checkpoint_{0}.tr")
        self.best_model_path = os.path.join(self.results_dir, "best_model.tr")
        self.plot_path = os.path.join(self.results_dir, "plot.png")

@dataclass
class VAEConfig:

    config_path : str = ""

    # files, directories and naming ---------------------------------------------
    delete :bool = True
    experiment_name :str = 'ssda'
    experiment_type :str = 'mnist'
    experiment_indentifier :str  = None
    init_model_path = None

    # ssda variables ------------------------------------------------------------
    z_dim: int = 20

    # all configs ---------------------------------------------------------------
    encoder: EncoderConfig = EncoderConfig()
    decoder: DecoderConfig = DecoderConfig()

    dataloader: NISTLoaderConfig = NISTLoaderConfig()
    trainer: VAETrainerConfig = VAETrainerConfig()
    experiment_files:VAE_ExperimentsFiles = None

    def __post_init__(self):
        self.experiment_files = VAE_ExperimentsFiles(delete=self.delete,
                                                     experiment_name=self.experiment_name,
                                                     experiment_indentifier=self.experiment_indentifier,
                                                     experiment_type=self.experiment_type)

        if isinstance(self.encoder, dict):
            self.encoder = all_encoders_configs[self.encoder["name"]](**self.encoder)
        if isinstance(self.decoder, dict):
            self.decoder = all_decoders_configs[self.decoder["name"]](**self.decoder)
        if isinstance(self.dataloader,dict):
            self.dataloader = all_dataloaders_configs[self.dataloader["name"]](**self.dataloader)
        if isinstance(self.trainer,dict):
            self.trainer = all_trainers_configs[self.trainer["name"]](**self.trainer)

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
        pass
        #self.encoder.input_size = self.dataloader.dimensions_per_variable * self.dataloader.number_of_variables

    def save_config(self):
        config_as_dict = asdict(self)
        with open(self.experiment_files.config_path, "w") as file:
            json.dump(config_as_dict, file)
