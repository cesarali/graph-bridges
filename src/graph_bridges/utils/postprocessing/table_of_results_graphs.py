import os
import time
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Union, Tuple, Dict,List
from graph_bridges.utils.postprocessing.table_of_results import TableOfResults
from graph_bridges.models.generative_models.ctdd import CTDD

from graph_bridges.data.graph_dataloaders_config import GraphDataConfig
from graph_bridges.configs.config_ctdd import CTDDConfig
from graph_bridges.configs.config_sb import SBConfig

from  graph_bridges.data.graph_dataloaders_config import (
    EgoConfig,
    CommunitySmallConfig,
    CommunityConfig,
    GridConfig
)

import numpy as np
import yaml

def get_experiment_dir(results_path,
                       experiment_name,
                       experiment_type,
                       experiment_indentifier=None):

    if experiment_indentifier is None:
        experiment_indentifier = str(int(time.time()))

    results_path = Path(results_path)

    all_experiments_dir = results_path / experiment_name
    experiment_type_dir = all_experiments_dir / experiment_type
    experiment_dir = experiment_type_dir / experiment_indentifier

    return (all_experiments_dir, experiment_type_dir, experiment_dir)


metrics_keys = ['initial_loss','LOSS','best_loss','training_loss','wasserstein','kl_distances','marginal_at_spins','training_time']

parameters_keys = ['dataloader_parameters',
                   'past_model_parameters',
                   'current_model_parameters',
                   'trainer_parameters',
                   'backward_estimator_parameters',
                   'sinkhorn_iteration',
                   'git_last_commit']

models_keys = ['current_model','past_model']

class GraphTablesOfResults(TableOfResults):
    """
    """
    def __init__(self,sinkhorn_to_read:int=0):

        datasets_names = ['Community', 'Ego','Grid']
        metrics_names = ['training_time','best_loss','MSE']
        methods_names = ['CTDD lr .001', "CTDD lr .01", "CTDD lr .05"]

        self.sinkhorn_to_read = sinkhorn_to_read

        TableOfResults.__init__(self,
                                "first_report",
                                datasets_names,
                                metrics_names,
                                methods_names,
                                False,
                                place_holder=np.inf)

    def dataset_name_to_config(self,dataset_name:str, config:Union[SBConfig,CTDDConfig]) -> Dict[int, Union[dict, dataclass]]:
        """

        :param dataset_name:
        :param config:
        :return: config
        """
        assert dataset_name in self.datasets_names
        if dataset_name == "Community":
            config.data = CommunitySmallConfig()
        elif dataset_name == "Ego":
            config.data = CommunitySmallConfig()
        elif dataset_name == "Grid":
            config.data = GridConfig()

        return config

    def metric_name_to_config(self) -> Dict[int, Union[dict, dataclass]]:
        pass

    def method_name_to_config(self) -> Dict[int, Union[dict, dataclass]]:
        pass

    def config_to_dataset_name(self,config:Union[SBConfig,CTDDConfig])->str:
        """
        :param config:
        :return:
        """
        def check_and_return(name_to_return):
            assert name_to_return in self.datasets_names
            return name_to_return

        if isinstance(config.data,EgoConfig) == "Ego":
            name_to_return = "Ego"
            assert name_to_return in self.datasets_names
            return name_to_return
        if isinstance(config.data,CommunitySmallConfig) == "Community":
            name_to_return = "Community"
            assert name_to_return in self.datasets_names
            return name_to_return
        if isinstance(config.data, GridConfig) == "Grid":
            name_to_return = "Grid"
            assert name_to_return in self.datasets_names
            return name_to_return
        return None

    def config_to_method_name(self,config:Union[SBConfig,CTDDConfig])->str:
        methods_names = ['CTDD lr .001', "CTDD lr .01", "CTDD lr .05"]
        def check_and_return(name_to_return):
            assert name_to_return in self.methods_names
            return name_to_return

        if config.trainer.learning_rate == 0.001:
            name_to_return = 'CTDD lr .001'
            check_and_return(name_to_return)
        elif config.trainer.learning_rate == 0.01:
            name_to_return = 'CTDD lr .01'
            check_and_return(name_to_return)
        elif config.trainer.learning_rate == 0.05:
            name_to_return = 'CTDD lr .05'
            check_and_return(name_to_return)
        return None

    def results_to_metrics(self,results_metrics:Dict)->Tuple[Dict[str,float],List[str]]:
        """
        metrics_names = ['training_time','best_loss','Degree','Cluster','Orbit']

        :param results_metrics:

        :return: metrics_in_file,missing_in_file
        """
        metrics_names = ['training_time','best_loss','MSE']

        metrics_in_file = {}
        missing_in_file = []

        for metric_to_check in self.metric_names:
            if metric_to_check in results_metrics.keys():
                metrics_in_file[metric_to_check] = results_metrics[metric_to_check]
            else:
                missing_in_file.append(metric_to_check)

        if "graphs_metrics" in results_metrics.keys():
            graphs_metrics = metrics_in_file["graphs_metrics"]
            metrics_in_file.update(graphs_metrics)
        else:
            missing_in_file.extend(['Degree','Cluster','Orbit'])

        return metrics_in_file,missing_in_file

    def experiment_dir_to_table(self,experiment_dir: Union[str, Path],overwrite=False,info=False):
        """
        modify table

        :param experiment_dir:

        :return: dataset_id,method_id,metrics_in_file,missing_in_file
        """
        configs,metrics,models,results = self.read_experiment_dir(experiment_dir)

        dataset_name = self.config_to_dataset_name(configs)
        methods_name = self.config_to_method_name(configs)
        metrics_in_file,missing_in_file = self.results_to_metrics(results)

        dataset_id = self.datasets_to_id[dataset_name]
        method_id = self.methods_to_id[methods_name]

        if info:
            print("Metrics found in {0}".format(experiment_dir))
            print(metrics_in_file)

        for key,new_value in metrics_in_file.items():
            if key in self.metrics_to_id.keys():
                metric_id = self.metrics_to_id[key]
                self.change_entry_id(dataset_id,metric_id,method_id,new_value,overwrite)

        return dataset_id,method_id,metrics_in_file,missing_in_file

    def experiment_dir_to_model(self, metric_name: str, experiment_dir: Union[str, Path]):
        """

        :param metric_name:
        :param experiment_dir:
        :return: dataset_name,method_name,metrics_in_file,missing_in_file,graph_diffusion_model
        """
        configs,metrics,models,results = self.read_experiment_dir(experiment_dir)
        graph_diffusion_model = GraphDiffusionModel(RESULTS=results)

        dataset_name = self.config_to_dataset_name(configs)
        method_name = self.config_to_method_name(configs)
        metrics_in_file,missing_in_file = self.results_to_metrics(results)

        return dataset_name,method_name,metrics_in_file,missing_in_file,graph_diffusion_model

    def run_config(self, config: Union[SBConfig,CTDDConfig]):
        if isinstance(config,CTDDConfig):
            from graph_bridges.models.trainers.ctdd_training import CTDDTrainer
            ctdd_trainer = CTDDTrainer(config)
            ctdd_trainer.train_ctdd()
        elif isinstance(config,SBConfig):
            from graph_bridges.models.trainers.sb_training import SBTrainer
            sb_trainer = SBTrainer(config)
            sb_trainer.train_schrodinger()

    def read_experiment_dir(self, experiment_dir: Union[str, Path]):
        """

        :param experiment_dir:
        :return: configs,metrics,models,results
        """
        from graph_bridges.configs.utils import get_config_from_file

        if isinstance(experiment_dir,str):
            experiment_dir = Path(experiment_dir)

        if experiment_dir.exists():

            config = get_config_from_file(experiment_dir)
            if isinstance(config,CTDDConfig):
                ctdd = CTDD()
                ctdd.load_from_results_folder(results_dir=experiment_dir)

            self.config_path = os.path.join(self.results_dir, "config.json")
            self.metrics_file = os.path.join(self.results_dir, "metrics_{0}.json")

            sinkhorn_to_read = "sinkhorn_{0}.tr".format(self.sinkhorn_to_read)
            results_path = experiment_dir / sinkhorn_to_read


            if results_path.exists():
                results = torch.load(results_path)

                configs = {key:results[key] for key in parameters_keys if key in results.keys()}
                metrics = {key:results[key] for key in metrics_keys if key in results.keys()}
                models = {key:results[key] for key in models_keys if key in results.keys()}

                return configs,metrics,models
            else:
                return None
        else:
            return None


if __name__=="__main__":
    from pprint import pprint
    from graph_bridges import results_path

    table_of_results = GraphTablesOfResults()

    #===================================================================
    # READ EXPERIMENT AND CHANGE TABLE
    #===================================================================

    _,_,experiment_dir = get_experiment_dir(results_path,
                                            experiment_name="graphs",
                                            experiment_type="",
                                            experiment_indentifier="lobster_to_efficient_one_1685024983")
    configs,metrics,models,results = table_of_results.read_experiment_dir(experiment_dir)

    print(table_of_results.config_to_dataset_name(configs))
    print(table_of_results.config_to_method_name(configs))

    dataset_id,method_id,metrics_in_file,missing_in_file = table_of_results.experiment_dir_to_table(experiment_dir,False,True)
    pprint(table_of_results.create_pandas())

    stuff = table_of_results.experiment_dir_to_model(None,experiment_dir)
    dataset_name, method_name, metrics_in_file, missing_in_file, graph_diffusion_model = stuff

    #===================================================================
    # DESIGN OF EXPERIMENTS
    #===================================================================

    from graph_bridges import project_path
    from graph_bridges.configs.graphs.graph_config_ctdd import CTDDConfig

    config = CTDDConfig()

    config = table_of_results.dataset_name_to_config("Community",config)
    table_of_results.run_config(config)

    #pprint(config)

