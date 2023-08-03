import os
import json
from typing import Union
from graph_bridges.configs.graphs.config_sb import SBConfig
from graph_bridges.configs.graphs.config_ctdd import CTDDConfig


def get_config_from_file(experiment_name,experiment_type,experiment_indentifier)->Union[SBConfig,CTDDConfig]:
    from graph_bridges import results_path

    experiment_dir = os.path.join(results_path, experiment_name)
    experiment_type_dir = os.path.join(experiment_dir, experiment_type)
    results_dir = os.path.join(experiment_type_dir, experiment_indentifier)

    config_path = os.path.join(results_dir, "config.json")
    config_path_json = json.load(open(config_path,"r"))
    config_path_json["delete"] = False

    config = SBConfig(**config_path_json)

    return config