import os
import time
import shutil
import subprocess
from pathlib import Path
from typing import Union,Tuple,List
from graph_bridges import results_path
from dataclasses import dataclass, asdict

def get_git_revisions_hash():
    hashes = []
    hashes.append(subprocess.check_output(['git', 'rev-parse', 'HEAD']))
    hashes.append(subprocess.check_output(['git', 'rev-parse', 'HEAD^']))
    return hashes

@dataclass
class ExperimentFiles:
    results_path:str = results_path
    experiment_indentifier:str = None
    experiment_name:str = None
    experiment_type:str= None
    results_dir:str = None

    data_stats:str = None
    delete:bool = False

    def __post_init__(self):
        self.results_path = str(results_path)
        self.current_git_commit = str(get_git_revisions_hash()[0])
        if self.experiment_indentifier is None:
            self.experiment_indentifier = str(int(time.time()))

        self.experiment_dir = os.path.join(self.results_path, self.experiment_name)
        self.experiment_type_dir = os.path.join(self.experiment_dir, self.experiment_type)
        self.results_dir = os.path.join(self.experiment_type_dir, self.experiment_indentifier)

        # doucet
        self.save_location = self.results_dir
        self.create_directories()

    def create_directories(self):
        if not Path(self.results_dir).exists():
            os.makedirs(self.results_dir)
        else:
            if self.delete:
                shutil.rmtree(self.results_dir)
                os.makedirs(self.results_dir)

        self.tensorboard_path = os.path.join(self.results_dir, "tensorboard")
        if os.path.isdir(self.tensorboard_path) and self.delete:
            shutil.rmtree(self.tensorboard_path)
        if not os.path.isdir(self.tensorboard_path):
            os.makedirs(self.tensorboard_path)

        self.config_path = os.path.join(self.results_dir, "config.json")
        self.metrics_file = os.path.join(self.results_dir, "metrics_{0}.json")


if __name__=="__main__":
    experiment_folders = ExperimentFiles(experiment_indentifier="test2",
                                         experiment_name="graph",
                                         experiment_type="ctdd")

