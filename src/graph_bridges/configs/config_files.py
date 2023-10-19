import os
import re
import time
import json
import torch
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

    project_results_dir:str = results_path
    experiment_indentifier:str = None
    experiment_name:str = None
    experiment_type:str= None
    results_dir:str = None

    data_stats:str = None
    delete:bool = False

    def __post_init__(self):
        if self.results_dir is None:
            self.project_results_dir = str(results_path)
            self.current_git_commit = str(get_git_revisions_hash()[0])
            if self.experiment_indentifier is None:
                self.experiment_indentifier = str(int(time.time()))

            self.experiment_dir = os.path.join(self.project_results_dir, self.experiment_name)
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

def create_experiment_dir(experiment_dir,experiment_name,experiment_type,experiment_indentifier):
    """
    if experiment_dir is None:
        experiment_dir := projects_results_dir/experiment_name/experiment_type/experiment_indentifier/

    :return:
    """
    if experiment_dir is None:
        projects_results_dir = str(results_path)
        if experiment_indentifier is None:
            experiment_indentifier = str(int(time.time()))

        experiment_name_dir = os.path.join(projects_results_dir, experiment_name)
        experiment_type_dir = os.path.join(experiment_name_dir, experiment_type)
        experiment_dir = os.path.join(experiment_type_dir, experiment_indentifier)

    return experiment_dir

@dataclass
class ExperimentFiles0:
    """
    if experiment_dir is None:
        experiment_dir := projects_results_dir/experiment_name/experiment_type/experiment_indentifier/

    """
    projects_results_dir:str = str(results_path)

    experiment_indentifier:str = None
    experiment_name:str = None
    experiment_type:str= None
    experiment_dir:str = None

    config_path:str = None
    best_model_path_checkpoint:str = None
    best_model_path:str = None
    metrics_file:str = None
    plot_path:str = None

    data_stats:str = None
    delete:bool = False

    def __post_init__(self):
        self.experiment_dir = create_experiment_dir(self.experiment_dir,
                                                    self.experiment_name,
                                                    self.experiment_type,
                                                    self.experiment_indentifier)

        self.current_git_commit = str(get_git_revisions_hash()[0])
        self.config_path = os.path.join(self.experiment_dir,"config.json")
        self.best_model_path_checkpoint = os.path.join(self.experiment_dir, "model_checkpoint_{0}.tr")
        self.best_model_path = os.path.join(self.experiment_dir, "best_model.tr")
        self.metrics_file = os.path.join(self.experiment_dir, "metrics_{0}.json")
        self.plot_path = os.path.join(self.experiment_dir, "plot_{0}.png")

    def create_directories(self):
        if not Path(self.experiment_dir).exists():
            os.makedirs(self.experiment_dir)
        else:
            if self.delete:
                shutil.rmtree(self.experiment_dir)
                os.makedirs(self.experiment_dir)
            else:
                raise Exception("Folder Exist no Experiments Created Set Delete to True")

        self.tensorboard_path = os.path.join(self.experiment_dir, "tensorboard")
        if os.path.isdir(self.tensorboard_path) and self.delete:
            shutil.rmtree(self.tensorboard_path)

        if not os.path.isdir(self.tensorboard_path):
            os.makedirs(self.tensorboard_path)


    #========================================================================================
    # READ FROM FILES
    #========================================================================================
    def load_metric(self, metric_string_identifier, checkpoint=None):
        def obtain_number (x):
            if x.name.split("_")[-1].split(".")[0].isdigit():
                return int(x.name.split("_")[-1].split(".")[0])
            else:
                return None

        generic_metric_path_ = self.metrics_file.format(metric_string_identifier + "*")
        generic_metric_path_to_fill = self.metrics_file.format(metric_string_identifier + "_{0}")
        generic_metric_path_ = Path(generic_metric_path_)

        # avaliable numbers
        numbers_available = []
        available_files = list(generic_metric_path_.parent.glob(generic_metric_path_.name))
        for file_ in available_files:
            numbers_available.append(obtain_number(file_))

        metrics_ = {}
        # read check point
        if checkpoint is not None:
            if checkpoint in numbers_available:
                metric_path_ = Path(generic_metric_path_to_fill.format(checkpoint))
                if metric_path_.exists():
                    metrics_ = json.load(open(metric_path_, "r"))

        if checkpoint is None:
            # read best file
            metric_path_ = Path(generic_metric_path_to_fill.format("best"))
            if metric_path_.exists():
                metrics_ = json.load(open(metric_path_, "r"))

            # read best available chackpoint
            else:
                if len(numbers_available) > 0:
                    best_number = max(numbers_available)
                    metric_path_ = Path(generic_metric_path_to_fill.format(best_number))
                    if metric_path_.exists():
                        metrics_ = json.load(open(metric_path_, "r"))
            return metrics_

    def load_all_metrics(self,all_metrics_identifiers,checkpoint):
        """
        ["graphs", "mse_histograms"]
        :param checkpoint:
        :return:
        """
        # SETS MODELS
        # READ METRICS IF AVAILABLE
        all_metrics = {}
        for metric_string_identifier in all_metrics_identifiers:
            all_metrics.update(self.load_metric(metric_string_identifier, checkpoint=checkpoint))
        return all_metrics

    def load_results(self, checkpoint=None, any=True):
        # LOADS RESULTS
        loaded_path = None
        if checkpoint is None:
            best_model_to_load_path = Path(self.best_model_path)

            if best_model_to_load_path.exists():
                results_ = torch.load(best_model_to_load_path)
                loaded_path = best_model_to_load_path
                return results_

            elif any:
                best_model_path_checkpoint = self.best_model_path_checkpoint
                extract_digits = lambda s: int(re.search(r'\d+', s).group()) if re.search(r'\d+', s) else None

                generic_metric_path_ = best_model_path_checkpoint.format("*")
                generic_metric_path_to_fill = best_model_path_checkpoint.format("{0}")
                generic_metric_path_ = Path(generic_metric_path_)

                # avaliable numbers
                numbers_available = []
                available_files = list(generic_metric_path_.parent.glob(generic_metric_path_.name))
                for file_ in available_files:
                    digits = extract_digits(str(file_.name))
                    numbers_available.append(digits)

                if len(numbers_available) > 0:
                    max_checkpoint = max(numbers_available)
                    generic_metric_path_to_fill = generic_metric_path_to_fill.format(max_checkpoint)
                    results_ = torch.load(generic_metric_path_to_fill)
                    loaded_path = generic_metric_path_to_fill
                    return  results_

        else:
            check_point_to_load_path = Path(self.best_model_path_checkpoint.format(checkpoint))
            if check_point_to_load_path.exists():
                results_ = torch.load(check_point_to_load_path)
                loaded_path = check_point_to_load_path
                return results_

        if loaded_path is None:
            print("Experiment Empty")
            return None

        # self.model = results_['current_model'].to(device)
        # SETS ALL OTHER CLASSES FROM CONFIG AND START NEW EXPERIMENT IF REQUIERED
        # self.config.align_configurations()
        # self.set_classes_from_config(self.config, device)



if __name__=="__main__":
    experiment_folders = ExperimentFiles(experiment_indentifier="test2",
                                         experiment_name="graph",
                                         experiment_type="ctdd")

