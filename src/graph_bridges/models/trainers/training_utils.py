import torch
from typing import Union
from graph_bridges.models.trainers.mutual_information_training import MITrainerConfig
from graph_bridges.models.trainers.mutual_information_training import BaseBinaryClassifier
from graph_bridges.models.trainers.mutual_information_training import MutualInformationConfig
from graph_bridges.models.trainers.mutual_information_training import BaseBinaryClassifierConfig
from graph_bridges.models.trainers.mutual_information_training import get_config_from_file

def load_binary_classifier(config:Union[BaseBinaryClassifierConfig,MutualInformationConfig]):
    if isinstance(config,MutualInformationConfig):
        config_ = config.binary_classifier
    elif isinstance(config, BaseBinaryClassifierConfig):
        config_ = config
    else:
        raise Exception("No Classifier Config Found")

    if config_.name == "BaseBinaryClassifier":
        binary_classifier = BaseBinaryClassifier(config_)
    else:
        raise Exception("No Classifier")
    return binary_classifier

def load_experiments_results(experiment_name, experiment_type, experiment_indentifier, checkpoint:int = None):
    config: MutualInformationConfig
    config = get_config_from_file(experiment_name, experiment_type, experiment_indentifier)
    if checkpoint is None:
        results = torch.load(config.experiment_files.best_model_path)
    else:
        results = torch.load(config.experiment_files.best_model_path_checkpoint.format(checkpoint))
    return config, results

def load_experiments_configuration(experiment_name, experiment_type, experiment_indentifier, checkpoint:int = None):
    from graph_bridges.data.mi_dataloader_utils import load_dataloader

    config: MutualInformationConfig
    config, results = load_experiments_results(experiment_name, experiment_type, experiment_indentifier, checkpoint)
    binary_classifier = results["binary_classifier"].to(torch.device("cpu"))
    dataloader = load_dataloader(config)

    return binary_classifier,dataloader