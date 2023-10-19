import os
import json
from typing import Union
from pathlib import Path
from graph_bridges.configs.config_sb import SBConfig
from graph_bridges.configs.config_ctdd import CTDDConfig
from graph_bridges.configs.config_oops import OopsConfig

def get_config_from_file(experiment_name=None,
                         experiment_type=None,
                         experiment_indentifier=None,
                         results_dir=None)->Union[SBConfig,CTDDConfig,OopsConfig]:
    if results_dir is None:
        from graph_bridges import results_path
        experiment_dir = os.path.join(results_path, experiment_name)
        experiment_type_dir = os.path.join(experiment_dir, experiment_type)
        results_dir = os.path.join(experiment_type_dir, experiment_indentifier)

    config_path = Path(os.path.join(results_dir, "config.json"))
    results_dir_as_path = Path(results_dir)

    if config_path.exists():
        config_path_json = json.load(open(config_path,"r"))
        config_path_json["delete"] = False
        config_path_json["config_path"] = str(config_path)
        config_path_json["experiment_files"]["results_dir"] = str(results_dir)

        if "loss" in config_path_json:
            if config_path_json["loss"]["name"] == "GenericAux":
                config = CTDDConfig(**config_path_json)
            elif config_path_json["loss"]["name"] == "BackwardRatioFlipEstimator":
                config = SBConfig(**config_path_json)
        else:
            if config_path_json["trainer"]["name"] == "ContrastiveDivergenceTrainer":
                config = OopsConfig(**config_path_json)
        return config
    else:
        return None