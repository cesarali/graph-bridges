import json

from typing import List,Union,Tuple,Optional,Dict
from matplotlib import pyplot as plt
from dataclasses import dataclass
from pathlib import Path
import torch
import time

import pandas as pd
import numpy as np
import subprocess
import os
from graph_bridges import results_path

from abc import ABC, abstractmethod

def get_git_revisions_hash():
    hashes = []
    hashes.append(subprocess.check_output(['git', 'rev-parse', 'HEAD']))
    hashes.append(subprocess.check_output(['git', 'rev-parse', 'HEAD^']))
    return hashes

def check_for_file(experiment_dir,file_to_check='sinkhorn_0.tr'):
    return file_to_check in os.listdir(experiment_dir)

class TableOfResults(ABC):
    """
    Abstract class that handles a table of results for a paper
    it has two hierarchies in the columns and one for the rows


                                  |           dataset1            |           dataset2            |     ...
             |    Methods/Results |   metric1     |   metric2     |   metric1     |   metric2     |
             |    method 1        |
    multirrow|    method 2        |       ***     |       ***     |       ***     |       ***     |
             |     ...
             |    method n        |       ***     |       ***     |       ***     |       ***     |


    each entry in this table is uniquely defined with

                        dataset_name:str, dataset_id:int,
                        metric_name:str , metric_id:int,
                        method_name:str , method_id:int,

    and one can modify a table entry with change_entry_id.

    The functionality aims at filling this table, we either:

    1.  Read results which are ready:

        from results of training AI models, typically we expect
        for every data experiment:

        1. a configuration file
        2. a results file
        3. a models file (results and models can coincide)
        4. metrics file

        The idea is to read results and configs and fill the tables one uses the abstract methods

    2. We can also generate config files that can be used to run experiments

        For especifics places of the table, in case we need to fill a
        particular entry

    3. Read models which are trained and perform the metrics requiered for fillling the table.

    """
    id_to_methods : dict
    id_to_datasets : dict
    id_to_metrics : dict
    table_data_frame : pd.DataFrame

    def __init__(self,
                 table_name:str,
                 datasets_names:List[str],
                 metric_names:List[str],
                 methods_names:List[str],
                 bigger_is_better: Union[bool, List] = True,
                 table_file: Union[str, Path] = None,
                 table_file_configs: Union[str, Path] = None,
                 data:Dict[Tuple[str,str],List[float]] = None,
                 place_holder:float=-np.inf,
                 multirowname:str="Real",
                 experiments_folder:Union[str,Path]=Path(results_path)):
        """

        :param table_name:
        :param datasets_names:
        :param metric_names:
        :param methods_names:
        :param bigger_is_better:
        :param table_file:
        :param data:
        :param place_holder:
        :param multirowname:
        :param experiments_folder:
        """
        #table names and files
        self.table_name = table_name
        self.experiments_folder = experiments_folder
        self.table_identifier = str(int(time.time()))

        if isinstance(self.experiments_folder,str):
            self.experiments_folder = Path(self.experiments_folder)

        if data is not None:
            self.data = data
        if table_file is not None:
            self.table_file = table_file
            self.table_file_configs = table_file_configs

        #table values
        self.number_of_methods = len(methods_names)
        self.number_of_datasets = len(datasets_names)
        self.number_of_metrics = len(metric_names)

        self.methods_names = methods_names
        self.metric_names = metric_names
        self.datasets_names = datasets_names
        self.place_holder = place_holder
        self.multirowname = multirowname

        if isinstance(bigger_is_better,bool):
            bigger_is_better = [bigger_is_better]*self.number_of_metrics
            self.bigger_is_better = bigger_is_better
        elif isinstance(bigger_is_better,list):
            assert len(bigger_is_better) == self.number_of_metrics
            self.bigger_is_better = bigger_is_better

        self.create_empty_table()

    def create_empty_table(self):
        empty_results = [self.place_holder] * self.number_of_methods

        self.id_to_datasets = {i: self.datasets_names[i] for i in range(self.number_of_datasets)}
        self.id_to_metrics = {i: self.metric_names[i] for i in range(self.number_of_metrics)}
        self.id_to_methods = {i: self.methods_names[i] for i in range(self.number_of_methods)}

        self.datasets_to_id = {self.datasets_names[i]:i for i in range(self.number_of_datasets)}
        self.metrics_to_id = {self.metric_names[i]:i for i in range(self.number_of_metrics)}
        self.methods_to_id = {self.methods_names[i]:i for i in range(self.number_of_methods)}

        data = {}
        for dataset_i in range(self.number_of_datasets):
            for results_j in range(self.number_of_metrics):
                data[(self.datasets_names[dataset_i], self.metric_names[results_j])] = empty_results[:]

        self.data = data

    def create_pandas(self):
        index = pd.MultiIndex.from_product([['Real'], self.methods_names])
        empty_results_dataframe = pd.DataFrame(self.data, index=index)
        return empty_results_dataframe

    def change_entry_names(self,
                           dataset_name:str,
                           metric_name:str,
                           method_name:str,
                           value:float,
                           overwrite:bool):

        dataset_id = self.datasets_to_id[dataset_name]
        metric_id = self.metrics_to_id[metric_name]
        method_id = self.methods_to_id[method_name]

        self.change_entry_id(dataset_id,metric_id,method_id,value,overwrite)

    def change_entry_id(self,
                        dataset_id:int,
                        metric_id:int,
                        method_id:int,
                        value:float,
                        overwrite:bool=False):

        if overwrite:
            self.data[(self.datasets_names[dataset_id], self.metric_names[metric_id])][method_id] = value
        else:
            current_value = self.data[(self.datasets_names[dataset_id], self.metric_names[metric_id])][method_id]
            bigger_is_better = self.bigger_is_better[metric_id]
            change = lambda current_value, value: value > current_value if bigger_is_better else value < current_value
            if change(current_value,value):
                all_row_values = self.data[(self.datasets_names[dataset_id], self.metric_names[metric_id])]
                all_row_values[method_id] = value
                self.data[(self.datasets_names[dataset_id], self.metric_names[metric_id])] = all_row_values

    def return_entry_names(self,
                           dataset_name:str,
                           metric_name:str,
                           method_name:str):
        dataset_id = self.datasets_to_id[dataset_name]
        metric_id = self.metrics_to_id[metric_name]
        method_id = self.methods_to_id[method_name]
        return self.return_entry_ids(dataset_id,metric_id,method_id)

    def return_entry_ids(self,
                         dataset_id:int,
                         metric_id:int,
                         method_id:int):
        return self.data[(self.datasets_names[dataset_id], self.metric_names[metric_id])][method_id]

    @abstractmethod
    def dataset_name_to_config(self,dataset_name,config)->Dict[int,Union[dict,dataclass]]:
        pass

    @abstractmethod
    def metric_name_to_config(self,metric_name,config)->Dict[int,Union[dict,dataclass]]:
        pass

    @abstractmethod
    def method_name_to_config(self,method_name,config)->Dict[int,Union[dict,dataclass]]:
        pass

    @abstractmethod
    def config_to_dataset_name(self,config)->str:
        """
        :param config:
        :return:
        """
        pass

    @abstractmethod
    def config_to_method_name(self,config)->str:
        pass

    @abstractmethod
    def results_to_metrics(self,config,results_,all_metrics)->Tuple[Dict[str,float],List[str]]:
        """
        metrics_names =

        :param results_metrics:

        :return: metrics_in_file,missing_in_file
        """
        pass

    @abstractmethod
    def read_experiment_dir(self,
                            experiment_dir:Union[Union[str,Path],List[Union[str,Path]]],
                            dataset_name:str = None,
                            metric_name:str = None,
                            method_name:str = None,
                            **kwargs):
        """
        this function loads the results from an experiment folder

        :return: sb,config,results,all_metrics,device
        """
        pass

    @abstractmethod
    def experiment_dir_to_model(self,
                                dataset_id:int,
                                metric_id:int,
                                method_id:int,
                                experiment_dir:Union[str,Path]):
        """
        if a trained model exist in the experiment folder, and the results file does not have the metrics
        generates the missing metric

        :param experiment_dir:
        :return:
        """
        pass

    @abstractmethod
    def run_config(self,config:Union[Dict,dataclass]):
        pass

    def run_table(self, base_methods_configs, base_dataset_args):
        for dataset_name in self.datasets_names:
            for method_name in self.methods_names:
                if method_name in base_methods_configs:

                    base_method_config = base_methods_configs[method_name]
                    base_method_config.experiment_indentifier = None
                    base_method_config.__post_init__()

                    base_method_config = self.dataset_name_to_config(dataset_name,base_method_config,base_dataset_args)

                    #set metrics
                    for metric_name in self.metric_names:
                        base_method_config = self.metric_name_to_config(metric_name,base_method_config)

                    #====================
                    # CHECK VALUE
                    #====================
                    current_value = self.return_entry_names(dataset_name=dataset_name,
                                                            metric_name=metric_name,
                                                            method_name=method_name)
                    #====================
                    # RUN CONFIG
                    #====================

                    self.run_config(base_method_config)

    def obtain_string_in_experiments(self,
                                     model_name,
                                     sinkhorn_iteration=0,
                                     strings_in_results=["training_time"]):
        """
        Checks in a results folder file for RESULTS with a given string

        :param model_name:
        :param sinkhorn_iteration:
        :param strings_in_results:
        :return:
        """
        from graph_bridges import results_path
        all_experiments_dir = os.path.join(results_path, model_name)
        all_experiments = os.listdir(all_experiments_dir)
        file_to_result = {}

        for experiment_dir_local in all_experiments:
            experiment_dir = os.path.join(all_experiments_dir, experiment_dir_local)
            if check_for_file(experiment_dir, 'sinkhorn_{0}.tr'.format(sinkhorn_iteration)):
                best_results_path = os.path.join(experiment_dir, "sinkhorn_{0}.tr".format(sinkhorn_iteration))

                RESULTS = torch.load(best_results_path)
                results_from_strings = {}
                for string_ in strings_in_results:
                    try:
                        results_ = RESULTS.get(string_)
                        results_from_strings[string_] = results_
                    except:
                        results_from_strings[string_] = None
            else:
                results_from_strings = False
            file_to_result[experiment_dir_local] = results_from_strings
        return file_to_result

    def save_table(self,save_dir:Union[str,Path]=None):
        if save_dir is None:
            save_dir = self.experiments_folder
        else:
            if isinstance(save_dir,str):
                save_dir = Path(save_dir)

        if save_dir.exists():
            table_file_name = self.table_name + "_" + self.table_identifier
            table_file = save_dir / table_file_name
            with open(table_file,"r") as file:
                json.dump(self.data,file)

    def read_table(self):
        pass

    def register_config_entry(self,data_id:int,method_id:int,config):
        return None