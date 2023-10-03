import os
import json
from typing import Union
from graph_bridges.configs.config_sb import SBConfig
from graph_bridges.configs.config_ctdd import CTDDConfig


def get_config_from_file(experiment_name=None,
                         experiment_type=None,
                         experiment_indentifier=None,
                         results_dir=None)->Union[SBConfig,CTDDConfig]:
    if results_dir is None:
        from graph_bridges import results_path
        experiment_dir = os.path.join(results_path, experiment_name)
        experiment_type_dir = os.path.join(experiment_dir, experiment_type)
        results_dir = os.path.join(experiment_type_dir, experiment_indentifier)

    config_path = os.path.join(results_dir, "config.json")
    config_path_json = json.load(open(config_path,"r"))
    config_path_json["delete"] = False

    if config_path_json["loss"]["name"] == "GenericAux":
        config = CTDDConfig(**config_path_json)
    elif config_path_json["loss"]["name"] == "BackwardRatioFlipEstimator":
        config = SBConfig(**config_path_json)

    return config