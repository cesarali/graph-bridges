import os
from itertools import product
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
import gc
from graph_bridges.models.metrics.sb_metrics_utils import log_metrics as log_sb_metrics
from graph_bridges.models.metrics.ctdd_metrics_utils import log_metrics as log_ctdd_metrics

from  graph_bridges.data.graph_dataloaders_config import (
    EgoConfig,
    CommunitySmallConfig,
    CommunityConfig,
    GridConfig
)

from typing import List
import numpy as np

from graph_bridges.models.generative_models.sb import SB
from dataclasses import dataclass
from graph_bridges.data.image_dataloader_config import NISTLoaderConfig

import yaml
import copy

import torch
from graph_bridges.utils.postprocessing.table_of_results_report_0 import TableOfResultsGraphBridges

def convert_paths_to_strings(data):
    if isinstance(data, dict):
        return {key: convert_paths_to_strings(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_paths_to_strings(item) for item in data]
    elif isinstance(data, Path):
        return str(data)
    else:
        return data

def time_inversion_combinations():
    input_string_to_format = "SB FLO {0} FLC {1} FP {2} FPS {3} FE {4}"

    # Generate all possible combinations of 1 and 0 for each entry
    combinations = list(product([0, 1], repeat=5))

    # Format the input string for each combination
    formatted_strings = [input_string_to_format.format(*combination) for combination in combinations]

    # Map name to combination
    string_to_combination = dict(zip(formatted_strings, combinations))

    return formatted_strings,string_to_combination

class TableOfResultsSchrodingerBridge(TableOfResultsGraphBridges):

    def __init__(self,
                table_name,
                datasets_names,
                metrics_names,
                methods_names,
                sinkhorn_to_read=0):


        #FLO  -> sb_config.loss.flip_old_time
        #FLC  -> sb_config.loss.flip_current_time
        #FP  -> sb_config.pipeline.flip_time
        #FPS  -> sb_config.pipeline.start_flip
        #FE  -> sb_config.pipeline.flip_even

        TableOfResultsGraphBridges.__init__(self,table_name,datasets_names,metrics_names,methods_names,sinkhorn_to_read)

        self.datasets_names_available.extend(['Community/Bernoulli'])

        self.formatted_strings, self.string_to_combination = time_inversion_combinations()
        self.methods_names_available.extend([self.formatted_strings])
        self.sinkhorn_to_read = sinkhorn_to_read


    def config_to_method_name(self,config:Union[SBConfig,CTDDConfig])->str:
        """

        :param config:
        :return:
        """
        super().config_to_method_name(config)
        FLO = int(config.loss.flip_old_time) if hasattr(config.loss,"flip_old_time") else None
        FLC = int(config.loss.flip_current_time) if hasattr(config.loss, "flip_old_time") else None
        FP = int(config.pipeline.flip_time) if hasattr(config.loss, "flip_old_time") else None
        FPS = int(config.pipeline.start_flip) if hasattr(config.loss, "flip_old_time") else None
        FE = int(config.pipeline.flip_even) if hasattr(config.loss, "flip_old_time") else None

        all_not_none = all(F is not None for F in [FLO, FLC, FP, FPS, FE])
        if all_not_none:
            input_string_to_format = "SB FLO {0} FLC {1} FP {2} FPS {3} FE {4}"
            method_name = input_string_to_format.format(FLO,FLC,FP,FPS,FE)

        if method_name in self.methods_names:
            return method_name

    def method_name_to_config(self, method_name, config: Union[SBConfig, CTDDConfig]) -> Dict[int, Union[dict, dataclass]]:
        """

        :param method_name:
        :param config:
        :return:
        """

        FLO,FLC,FP,FPS,FE = tuple(self.string_to_combination[method_name])

        config.loss.flip_old_time = bool(FLO)
        config.loss.flip_current_time = bool(FLC)
        config.pipeline.flip_time = FP
        config.pipeline.start_flip = bool(FPS)
        config.pipeline.flip_even = bool(FE)

        return config

    def run_config(self, new_config: Union[SBConfig,CTDDConfig]):

        if isinstance(new_config,SBConfig):
            from graph_bridges.models.trainers.sb_training import SBTrainer

            sb_trainer = SBTrainer(config=None,
                                   experiment_name="graph",
                                   experiment_type="sb",
                                   experiment_indentifier="community_small_to_bernoulli",
                                   new_experiment_indentifier=None,
                                   sinkhorn_iteration_to_load=0,
                                   next_sinkhorn=True)

            # adjust loss
            sb_trainer.sb.backward_ratio_estimator.flip_current_time = new_config.loss.flip_current_time
            sb_trainer.sb.backward_ratio_estimator.flip_old_time = new_config.loss.flip_old_time
            sb_trainer.sb.config.loss = new_config.loss
            sb_trainer.sb_config.loss = new_config.loss

            # adjust pipeline
            sb_trainer.sb.pipeline.start_flip = new_config.pipeline.start_flip
            sb_trainer.sb.pipeline.flip_time = new_config.pipeline.flip_time
            sb_trainer.sb.pipeline.flip_even = new_config.pipeline.flip_even

            sb_trainer.sb.config.pipeline = new_config.pipeline
            sb_trainer.sb_config.pipeline = new_config.pipeline

            # adjust training
            sb_trainer.sb_config.trainer.metrics = new_config.trainer.metrics
            sb_trainer.number_of_epochs = new_config.trainer.num_epochs
            sb_trainer.sb_config.trainer.save_metric_epochs = int(.25*sb_trainer.number_of_epochs)
            sb_trainer.sb_config.trainer.save_model_epochs = int(.25*sb_trainer.number_of_epochs)

            sb_trainer.sb_config.sampler.num_steps = new_config.sampler.num_steps

            # save config again
            sb_trainer.sb_config.save_config()
            results_,all_metrics = sb_trainer.train_schrodinger()

            results_["current_model"] = None
            results_["past_model"] = None
            torch.cuda.empty_cache()
            gc.collect()

            new_config.experiment_files.results_dir = sb_trainer.sb.config.experiment_files.results_dir

        return results_,all_metrics

    def read_and_log_new_metrics(self,path_of_model, metrics_names):
        read_results = self.read_experiment_dir(path_of_model)
        if read_results is not None:
            sb, config, results, all_metrics, device = read_results
            if isinstance(config,SBConfig):
                from graph_bridges.models.pipelines.sb.pipeline_sb import SBPipeline
                from graph_bridges.utils.test_utils import check_model_devices

                sb.pipeline : SBPipeline
                method_name = self.config_to_method_name(config)

                print(method_name)
                resersed_time_file = config.experiment_files.plot_path.format("reversed_time")

                # FP  -> sb_config.pipeline.flip_time
                # FPS  -> sb_config.pipeline.start_flip
                # FE  -> sb_config.pipeline.flip_even

                sb.pipeline.flip_time = not(sb.pipeline.flip_time)
                sb.pipeline.start_flip = 0
                sb.pipeline.flip_even = not(sb.pipeline.flip_even)

                training_model = sb.training_model
                past_to_train_model = sb.past_model

                paths,time = sb.pipeline(training_model,1)

                log_sb_metrics(sb=sb,
                               current_model=training_model,
                               past_to_train_model=past_to_train_model,
                               sinkhorn_iteration=1,
                               device=check_model_devices(past_to_train_model),
                               epoch=None,
                               metrics_to_log=metrics_names,
                               where_to_log={"histograms":resersed_time_file})

            elif isinstance(config, CTDDConfig):
                    print("Hey You!")

        return None

if __name__=="__main__":
    # ===================================================================
    # DEFINE THE TABLE
    # ===================================================================

    from pprint import pprint

    formatted_strings, string_to_combination = time_inversion_combinations()

    datasets_names_ = ['Community']
    metrics_names_ = ['Best Loss', 'MSE']
    methods_names_ = formatted_strings[-10:-1]

    table_of_results = TableOfResultsSchrodingerBridge(table_name="back_sb",
                                                       datasets_names=datasets_names_,
                                                       metrics_names=metrics_names_,
                                                       methods_names=methods_names_)
    pandas_table = table_of_results.create_pandas()
    pprint(pandas_table)

    # DEFINE BASE CONFIGS
    sb_config = SBConfig(experiment_name="back_sb",
                         experiment_type="table_report_0")

    sb_config.trainer.metrics = ["histograms"]
    sb_config.trainer.num_epochs = 4
    sb_config.trainer.__post_init__()
    sb_config.sampler.num_steps = 4

    #base methods and datasets arguments
    base_methods_configs = {method_name: copy.deepcopy(sb_config) for method_name in formatted_strings}
    base_dataset_args = {"batch_size":20}

    #table_of_results.run_table(base_methods_configs,base_dataset_args)

    #=========================================
    # READ RESULTS
    #=========================================

    parent_experiment_folder = "C:/Users/cesar/Desktop/Projects/DiffusiveGenerativeModelling/Codes/graph-bridges/results/graph/sb"
    parent_experiment_folder_porous = ""

    table_of_results.sinkhorn_to_read = 1
    table_of_results.fill_table([parent_experiment_folder],info=True)

    print("Final Table")
    pandas_table = table_of_results.create_pandas()
    pprint(pandas_table)
    files_pandas_table = table_of_results.create_files_pandas()
    pprint(files_pandas_table)

    #table_of_results.log_new_metrics("MSE",["histograms"])