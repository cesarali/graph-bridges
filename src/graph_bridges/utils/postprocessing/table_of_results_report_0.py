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
from graph_bridges.models.generative_models.sb import SB
from dataclasses import dataclass
import copy

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

@dataclass
class BaseConfig:

    base_ctdd : CTDDConfig = None
    base_sb: SBConfig = None
    base_dataset_args: dict = None
    base_method_args: dict = None

class TableOfResultsReport(TableOfResults):
    """
    """
    def __init__(self,sinkhorn_to_read:int=0):

        datasets_names = ['Community','Ego','Grid']

        metrics_names = ['Best Loss','MSE']

        methods_names = ['CTDD lr .001', "CTDD lr .01", "CTDD lr .05"]

        self.sinkhorn_to_read = sinkhorn_to_read

        TableOfResults.__init__(self,
                                "first_report",
                                datasets_names,
                                metrics_names,
                                methods_names,
                                False,
                                place_holder=np.inf)

    #============================================================
    # MAPPING TO TABLE
    #============================================================
    def dataset_name_to_config(self,dataset_name:str, config:Union[SBConfig,CTDDConfig],base_dataset_args:dict={}) -> Dict[int, Union[dict, dataclass]]:
        """

        :param dataset_name:
        :param config:
        :return: config
        """
        assert dataset_name in self.datasets_names

        if dataset_name == "Community":
            config.data = CommunitySmallConfig(**base_dataset_args)
        elif dataset_name == "Ego":
            config.data = EgoConfig(**base_dataset_args)
        elif dataset_name == "Grid":
            config.data = GridConfig(**base_dataset_args)

        return config

    def metric_name_to_config(self,metric_name,config:Union[SBConfig,CTDDConfig]) -> Dict[int, Union[dict, dataclass]]:
        if metric_name == 'MSE':
            metric_name = "mse_histograms"
            if not metric_name in config.trainer.metrics:
                config.trainer.metrics.append("mse_histograms")
        return config

    def method_name_to_config(self,method_name,config:Union[SBConfig,CTDDConfig]) -> Dict[int, Union[dict, dataclass]]:
        if method_name == 'CTDD lr .001':
            config.trainer.learning_rate = .001
        if method_name == "CTDD lr .01":
            config.trainer.learning_rate = .01
        if method_name == "CTDD lr .05":
            config.trainer.learning_rate = .05
        return config

    def config_to_dataset_name(self,config:Union[SBConfig,CTDDConfig])->str:
        """
        :param config:
        :return:
        """
        if isinstance(config.data,EgoConfig):
            name_to_return = "Ego"
            assert name_to_return in self.datasets_names
            return name_to_return
        if isinstance(config.data,CommunitySmallConfig):
            name_to_return = "Community"
            assert name_to_return in self.datasets_names
            return name_to_return
        if isinstance(config.data, GridConfig):
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

    def results_to_metrics(self,config:Union[SBConfig,CTDDConfig],results_,all_metrics:Dict)->Tuple[Dict[str,float],List[str]]:
        """
        Parse results and metrics for the requiered values

        :param results_metrics:
        :return:
        """
        missing_in_file = []
        metrics_in_file = {}

        if "best_loss" in results_:
            metrics_in_file['Best Loss'] = results_["best_loss"]
        else:
            missing_in_file.append('Best Loss')

        if "mse_histograms_0" in all_metrics:
            metrics_in_file["MSE"] = all_metrics["mse_histograms_0"]
        else:
            missing_in_file.append("MSE")

        return metrics_in_file,missing_in_file

    #===============================================================
    # TABLE STUFF
    #===============================================================
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
                results,all_metrics,device = ctdd.load_from_results_folder(results_dir=experiment_dir)
                return ctdd,config,results,all_metrics,device
            elif isinstance(config,SBConfig):
                sb = SB()
                results,all_metrics,device = sb.load_from_results_folder(results_dir=experiment_dir)
                return sb,config,results,all_metrics,device
        else:
            return None

    def experiment_dir_to_table(self,experiment_dir: Union[str, Path],overwrite=False,info=False):
        """
        modify table

        :param experiment_dir:

        :return: dataset_id,method_id,metrics_in_file,missing_in_file
        """
        results_of_reading = self.read_experiment_dir(experiment_dir)
        if results_of_reading is not None:
            generative_model,configs, results, all_metrics, device = results_of_reading

            dataset_name = self.config_to_dataset_name(configs)
            methods_name = self.config_to_method_name(configs)
            metrics_in_file,missing_in_file = self.results_to_metrics(results,all_metrics)

            dataset_id = self.datasets_to_id[dataset_name]
            method_id = self.methods_to_id[methods_name]

            if info:
                print("Metrics found in {0}".format(experiment_dir))
                print(metrics_in_file)

            for key,new_value in metrics_in_file.items():
                metrics_to_id_keys = self.metrics_to_id.keys()
                if key in metrics_to_id_keys:
                    metric_id = self.metrics_to_id[key]
                    self.change_entry_id(dataset_id,metric_id,method_id,new_value,overwrite)

            return dataset_id,method_id,metrics_in_file,missing_in_file

    def experiment_dir_to_model(self, metric_name: str, experiment_dir: Union[str, Path]):
        """

        :param metric_name:
        :param experiment_dir:
        :return: dataset_name,method_name,metrics_in_file,missing_in_file,graph_diffusion_model
        """
        sb,config,results,all_metrics,device = self.read_experiment_dir(experiment_dir)

        dataset_name = self.config_to_dataset_name(config)
        method_name = self.config_to_method_name(config)
        metrics_in_file,missing_in_file = self.results_to_metrics(results,all_metrics)

        return sb

    #===============================================================
    # RUN TABLE
    #===============================================================
    def set_baselines(self,base_config:BaseConfig):
        self.base_method_config = base_config.base_ctdd
        self.base_sb_config = base_config.base_sb

        self.base_dataset_args = base_config.base_dataset_args
        self.base_method_args = base_config.base_method_args

    def run_config(self, config: Union[SBConfig,CTDDConfig]):
        if isinstance(config,CTDDConfig):
            from graph_bridges.models.trainers.ctdd_training import CTDDTrainer
            ctdd_trainer = CTDDTrainer(config)
            ctdd_trainer.train_ctdd()
        elif isinstance(config,SBConfig):
            from graph_bridges.models.trainers.sb_training import SBTrainer
            sb_trainer = SBTrainer(config)
            sb_trainer.train_schrodinger()


if __name__=="__main__":
    from pprint import pprint

    # ===================================================================
    # JUST READ THE TABLE
    # ===================================================================

    table_of_results = TableOfResultsReport()
    pandas_table = table_of_results.create_pandas()

    # ===================================================================
    # DESIGN OF EXPERIMENTS
    # ===================================================================

    from graph_bridges.configs.graphs.graph_config_ctdd import CTDDConfig
    from graph_bridges.configs.graphs.graph_config_sb import SBConfig

    #datasets_names = ['Community', 'Ego', 'Grid']
    #metrics_names = ['Best Loss', 'MSE']
    #methods_names = ['CTDD lr .001', "CTDD lr .01", "CTDD lr .05"]

    config = CTDDConfig(experiment_name="table",
                        experiment_type="table_report_0",
                        delete=True)
    config.trainer.metrics = []
    config.trainer.num_epochs = 5
    config.trainer.__post_init__()
    config = table_of_results.dataset_name_to_config("Community", config)

    base_methods_configs = {'CTDD lr .001':copy.deepcopy(config),
                            "CTDD lr .01":copy.deepcopy(config),
                            "CTDD lr .05":copy.deepcopy(config)}

    base_dataset_args = {"batch_size":32,"full_adjacency":False}

    table_of_results.run_table(base_methods_configs,base_dataset_args)
    #table_of_results.run_config(config)

    """
    
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
    
        #pprint(config)
        
    """