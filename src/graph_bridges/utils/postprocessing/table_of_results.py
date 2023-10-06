import sys
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
from graph_bridges.configs.config_sb import SBConfig
from graph_bridges.configs.config_ctdd import CTDDConfig
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
                 bigger_is_better: Union[bool, List] = False,
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
        empty_files = [None] * self.number_of_methods

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

        files_names = {}
        for dataset_i in range(self.number_of_datasets):
            for results_j in range(self.number_of_metrics):
                files_names[(self.datasets_names[dataset_i], self.metric_names[results_j])] = empty_files[:]

        files_paths = {}
        for dataset_i in range(self.number_of_datasets):
            for results_j in range(self.number_of_metrics):
                files_paths[(self.datasets_names[dataset_i], self.metric_names[results_j])] = empty_files[:]

        self.data = data
        self.files_names = files_names
        self.files_paths = files_paths


    def create_pandas(self):
        index = pd.MultiIndex.from_product([['Real'], self.methods_names])
        empty_results_dataframe = pd.DataFrame(self.data, index=index)
        return empty_results_dataframe

    def create_files_pandas(self):
        index = pd.MultiIndex.from_product([['Real'], self.methods_names])
        empty_results_dataframe = pd.DataFrame(self.files_names, index=index)
        return empty_results_dataframe


    def change_entry_names(self,
                           dataset_name:str,
                           metric_name:str,
                           method_name:str,
                           value:float,
                           overwrite:bool,
                           where_is_it:str=None):

        dataset_id = self.datasets_to_id[dataset_name]
        metric_id = self.metrics_to_id[metric_name]
        method_id = self.methods_to_id[method_name]

        self.change_entry_id(dataset_id,metric_id,method_id,value,overwrite,where_is_it)

    def change_entry_id(self,
                        dataset_id:int,
                        metric_id:int,
                        method_id:int,
                        value:float,
                        overwrite:bool=False,
                        where_is_it:str=None):

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

                if where_is_it is not None:
                    if isinstance(where_is_it,str):
                        where_is_it = Path(where_is_it)
                    if isinstance(where_is_it,Path):
                        all_row_values_names = self.files_names[(self.datasets_names[dataset_id], self.metric_names[metric_id])]
                        all_row_values_names[method_id] = where_is_it.name
                        all_row_values_paths = self.files_paths[(self.datasets_names[dataset_id], self.metric_names[metric_id])]
                        all_row_values_paths[method_id] = where_is_it

                        self.files_names[(self.datasets_names[dataset_id], self.metric_names[metric_id])] = all_row_values_names
                        self.files_paths[(self.datasets_names[dataset_id], self.metric_names[metric_id])] = all_row_values_paths

    def entry_from_names(self,
                         dataset_name:str,
                         metric_name:str,
                         method_name:str):
        dataset_id = self.datasets_to_id[dataset_name]
        metric_id = self.metrics_to_id[metric_name]
        method_id = self.methods_to_id[method_name]
        return self.entry_from_id(dataset_id, metric_id, method_id)

    def path_from_names(self,
                        dataset_name: str,
                        metric_name: str,
                        method_name: str):
        dataset_id = self.datasets_to_id[dataset_name]
        metric_id = self.metrics_to_id[metric_name]
        method_id = self.methods_to_id[method_name]
        return self.path_from_ids(dataset_id, metric_id, method_id)

    def entry_from_id(self,
                      dataset_id:int,
                      metric_id:int,
                      method_id:int):
        return self.data[(self.datasets_names[dataset_id], self.metric_names[metric_id])][method_id]

    def path_from_ids(self,
                      dataset_id: int,
                      metric_id: int,
                      method_id: int):
        return self.files_paths[(self.datasets_names[dataset_id], self.metric_names[metric_id])][method_id]

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

    #=================================================================
    # READ AND CHANGE ENTRIES
    #=================================================================

    def fill_table(self,base_folder,overwrite=False,info=False):
        if isinstance(base_folder,str):
            base_folder = [base_folder]
        for base_folder_ in base_folder:
            base_path = Path(base_folder_)
            # Iterate through all subfolders
            for subfolder in base_path.iterdir():
                if subfolder.is_dir():
                    print(subfolder)
                    self.experiment_dir_to_table(subfolder,overwrite,info)


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
            metrics_in_file,missing_in_file = self.results_to_metrics(configs,results,all_metrics)

            if dataset_name is not None and methods_name is not None:
                dataset_id = self.datasets_to_id[dataset_name]
                method_id = self.methods_to_id[methods_name]

                if info:
                    print("Metrics found in {0}".format(experiment_dir))
                    print(metrics_in_file)
                    results_of_reading = self.read_experiment_dir(experiment_dir)

                for key,new_posible_value in metrics_in_file.items():
                    metrics_to_id_keys = self.metrics_to_id.keys()
                    if key in metrics_to_id_keys:
                        metric_id = self.metrics_to_id[key]
                        if isinstance(new_posible_value, torch.Tensor):
                            new_posible_value = new_posible_value.cpu().item()
                        self.change_entry_id(dataset_id,metric_id,method_id,new_posible_value,overwrite,experiment_dir)

                return dataset_id,method_id,metrics_in_file,missing_in_file

    #=================================================================
    # RUNNING EXPERIMENTS
    #=================================================================

    @abstractmethod
    def run_config(self,config:Union[Dict,dataclass]):
        pass

    def run_table(self, base_methods_configs, base_dataset_args,fill_table=True):
        """
        Using base configuration files run experiments necesary to fill the table

        :param base_methods_configs:
        :param base_dataset_args:
        :param fill_table:
        :return:
        """
        for dataset_name in self.datasets_names:
            for method_name in self.methods_names:
                if method_name in base_methods_configs:

                    print("MEMORY INFORMATION")
                    free, total = torch.cuda.mem_get_info()
                    free_mb = round(free / 1024 ** 2,2)
                    total_mb = round(total / 1024 ** 2,2)
                    print(f'Free: {free_mb} MB, Total: {total_mb} MB')

                    base_method_config : Union[SBConfig,CTDDConfig]
                    base_method_config = base_methods_configs[method_name]
                    base_method_config.experiment_indentifier = None
                    base_method_config.__post_init__()

                    base_method_config = self.dataset_name_to_config(dataset_name,base_method_config,base_dataset_args)
                    base_method_config = self.method_name_to_config(method_name,base_method_config)

                    #set metrics
                    for metric_name in self.metric_names:
                        base_method_config = self.metric_name_to_config(metric_name,base_method_config)

                    #current_value = self.return_entry_names(dataset_name=dataset_name,
                    #                                        metric_name=metric_name,
                    #                                        method_name=method_name)

                    #====================
                    # RUN CONFIG
                    #====================

                    try:
                        results, all_metrics = self.run_config(base_method_config)
                    except Exception as e:
                        results = None
                        all_metrics = None
                        print(f"An error occurred: {e}")
                        print(f"Exception details: {sys.exc_info()}")

                    if fill_table and results is not None:
                        metrics_in_file, missing_ = self.results_to_metrics(base_method_config,results, all_metrics)

                        for metric_name_ in metrics_in_file:
                            new_posible_value = metrics_in_file[metric_name_]
                            if isinstance(new_posible_value,torch.Tensor):
                                new_posible_value = new_posible_value.cpu().item()
                            self.change_entry_names(dataset_name=dataset_name,
                                                    metric_name=metric_name_,
                                                    method_name=method_name,
                                                    value=new_posible_value,
                                                    overwrite=False,
                                                    where_is_it=base_method_config.experiment_files.results_dir)

    #====================================================================
    # LOG NEW METRICS
    #====================================================================
    def log_new_metrics(self,metric_selector,metrics_names):
        if metric_selector in self.metric_names:
            for dataset_name in self.datasets_names:
                for method_name in self.methods_names:
                    path_of_model = self.path_from_names(dataset_name=dataset_name,metric_name=metric_selector,method_name=method_name)
                    if isinstance(path_of_model,str):
                        path_of_model = Path(path_of_model)
                    self.read_and_log_new_metrics(path_of_model,metrics_names)
    @abstractmethod
    def read_and_log_new_metrics(self,path_of_model,metrics_names):
        """
        to the model located in path of model
        :param path_of_model:
        :param metrics_names:
        :return:
        """
        return None

    #====================================================================
    # SAVE AND READ TABLE
    #====================================================================

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

