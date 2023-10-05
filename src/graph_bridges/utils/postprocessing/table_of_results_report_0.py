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

from typing import List
import numpy as np

from graph_bridges.models.generative_models.sb import SB
from dataclasses import dataclass
from graph_bridges.data.image_dataloader_config import NISTLoaderConfig

import yaml
import copy

class TableOfResultsGraphBridges(TableOfResults):
    """
    """
    def __init__(self,
                 table_name,
                 datasets_names,
                 metrics_names,
                 methods_names,
                 sinkhorn_to_read:int=0):

        self.datasets_names_available = ['Community','Ego','Grid','MNIST','EMNIST',"Fashion"]
        self.metrics_names_available = ['Best Loss','MSE']
        self.methods_names_available = ["CTDD lr .05",'CTDD lr .001', "CTDD lr .01","SB lr: 0.01","SB lr: 0.007"]

        self.sinkhorn_to_read = sinkhorn_to_read

        TableOfResults.__init__(self,
                                table_name,
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
        :param base_dataset_args: arguments shared across data specifications
        :return: config
        """
        #'MNIST', 'EMNIST', "Fashion"
        assert dataset_name in self.datasets_names

        if dataset_name == "Community":
            config.data = CommunitySmallConfig(**base_dataset_args)
        elif dataset_name == "Ego":
            config.data = EgoConfig(**base_dataset_args)
        elif dataset_name == "Grid":
            config.data = GridConfig(**base_dataset_args)
        elif dataset_name == "MNIST":
            config.data = NISTLoaderConfig(**base_dataset_args)
        elif dataset_name == "EMNIST":
            config.data = NISTLoaderConfig(**base_dataset_args)
        elif dataset_name == "Fashion":
            config.data = NISTLoaderConfig(**base_dataset_args)

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
            #config.trainer.num_epochs = 150
        if method_name == "CTDD lr .01":
            config.trainer.learning_rate = .01
            #config.trainer.num_epochs = 50
            config.trainer.__post_init__()
        if method_name == "CTDD lr .05":
            config.trainer.learning_rate = .05
        if method_name == "SB lr: 0.01":
            assert isinstance(config,SBConfig)
            config.trainer.learning_rate = .01
        if method_name == "SB lr: 0.007":
            assert isinstance(config,SBConfig)
            config.trainer.learning_rate = .007
        return config

    def config_to_dataset_name(self,config:Union[SBConfig,CTDDConfig])->str:
        """
        :param config:
        :return:
        """
        if isinstance(config.data,EgoConfig):
            name_to_return = "Ego"
            return name_to_return if name_to_return in self.datasets_names else None
        if isinstance(config.data,CommunitySmallConfig):
            name_to_return = "Community"
            return name_to_return if name_to_return in self.datasets_names else None
        if isinstance(config.data, GridConfig):
            name_to_return = "Grid"
            return name_to_return if name_to_return in self.datasets_names else None
        if isinstance(config.data, NISTLoaderConfig):
            if config.data.data == "mnist":
                name_to_return = "MNIST"
            if config.data.data == "emnist":
                name_to_return = "EMNIST"
            if config.data.data == "fashion":
                name_to_return = "Fashion"
            return name_to_return if name_to_return in self.datasets_names else None

    def config_to_method_name(self,config:Union[SBConfig,CTDDConfig])->str:

        def check_and_return(name_to_return):
            return name_to_return if name_to_return in self.methods_names else None

        #===============================
        # CTDD EXPERIMENTS CHECK
        #===============================
        if isinstance(config,CTDDConfig):
            if config.trainer.learning_rate == 0.001:
                name_to_return = 'CTDD lr .001'
                return check_and_return(name_to_return)
            elif config.trainer.learning_rate == 0.01:
                name_to_return = 'CTDD lr .01'
                return check_and_return(name_to_return)
            elif config.trainer.learning_rate == 0.05:
                name_to_return = 'CTDD lr .05'
                return check_and_return(name_to_return)

        #===============================
        # SB EXPERIMENTS CHECK
        #===============================
        if isinstance(config,SBConfig):
            if config.trainer.learning_rate == 0.01:
                name_to_return = "SB lr: 0.01"
                return check_and_return(name_to_return)
            elif config.trainer.learning_rate == 0.007:
                name_to_return = "SB lr: 0.007"
                if name_to_return in self.methods_names:
                    return name_to_return

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
    # FILLING STUFF
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
            config = get_config_from_file(results_dir=experiment_dir)
            if isinstance(config,CTDDConfig):
                ctdd = CTDD()
                all_results = ctdd.load_from_results_folder(results_dir=experiment_dir)
                if all_results is not None:
                    results,all_metrics,device = all_results
                    return ctdd,config,results,all_metrics,device
                else:
                    return None
            elif isinstance(config,SBConfig):
                sb = SB()
                all_results = sb.load_from_results_folder(results_dir=experiment_dir,
                                                          sinkhorn_iteration_to_load=self.sinkhorn_to_read,
                                                          any=True)
                if all_results is not None:
                    results, all_metrics, device = all_results
                    return sb,config,results,all_metrics,device
                else:
                    return None
        else:
            return None

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
    # RUNNIG TABLE
    #===============================================================

    def run_config(self, config: Union[SBConfig,CTDDConfig]):
        if isinstance(config,CTDDConfig):
            from graph_bridges.models.trainers.ctdd_training import CTDDTrainer
            ctdd_trainer = CTDDTrainer(config)
            results_,all_metrics = ctdd_trainer.train_ctdd()
        elif isinstance(config,SBConfig):
            from graph_bridges.models.trainers.sb_training import SBTrainer
            sb_trainer = SBTrainer(config)
            results_,all_metrics = sb_trainer.train_schrodinger()
        return results_,all_metrics


if __name__=="__main__":
    from pprint import pprint

    # ===================================================================
    # JUST READ THE TABLE
    # ===================================================================

    datasets_names_ = ['MNIST', 'EMNIST', "Fashion"]
    metrics_names_ = ['Best Loss', 'MSE']
    methods_names_ = ["CTDD lr .05", 'CTDD lr .001', "CTDD lr .01"]

    table_of_results = TableOfResultsGraphBridges(table_name="Nist_CTDD",
                                                  datasets_names=datasets_names_,
                                                  metrics_names=metrics_names_,
                                                  methods_names=methods_names_)
    pandas_table = table_of_results.create_pandas()

    pprint(pandas_table)

    # ===================================================================
    # DESIGN OF EXPERIMENTS
    # ===================================================================

    from graph_bridges.configs.graphs.graph_config_ctdd import CTDDConfig as GraphCTDD
    from graph_bridges.configs.images.nist_config_ctdd import CTDDConfig as NistCTDD
    from graph_bridges.configs.graphs.graph_config_sb import SBConfig

    from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig

    #datasets_names = ['Community', 'Ego', 'Grid']
    #metrics_names = ['Best Loss', 'MSE']
    #methods_names = ['CTDD lr .001', "CTDD lr .01", "CTDD lr .05", "SB lr: 0.01"]

    config = NistCTDD(experiment_name="NIST",
                      experiment_type="table_report_more_learning",
                      delete=True)

    config.model = BackRateMLPConfig(hidden_layer=350,time_embed_dim=128)

    config.trainer.metrics = ["histograms"]
    config.trainer.num_epochs = 20
    config.trainer.__post_init__()

    sb_config = SBConfig(experiment_name="table",
                         experiment_type="table_report_0",
                         delete=True)
    
    sb_config.trainer.metrics = []
    sb_config.trainer.num_epochs = 5
    sb_config.trainer.__post_init__()

    sb_config.sampler.num_steps = 5
    sb_config.flip_estimator.stein_sample_size = 200

    base_methods_configs = {'CTDD lr .001':copy.deepcopy(config),
                            "CTDD lr .01":copy.deepcopy(config),
                            "CTDD lr .05":copy.deepcopy(config)}

    #base_dataset_args = {"batch_size":32,"full_adjacency":False}
    base_dataset_args = {"batch_size":128}
    table_of_results.run_table(base_methods_configs,base_dataset_args)

    print("Final Table")
    pandas_table = table_of_results.create_pandas()
    pprint(pandas_table)
    

    # ===================================================================
    # READ EXPERIMENT AND CHANGE TABLE
    # ===================================================================
    """
    parent_experiment_folder = "C:/Users/cesar/Desktop/Projects/DiffusiveGenerativeModelling/Codes"
    parent_experiment_folder2 = "C:/Users/cesar/Desktop/Projects/DiffusiveGenerativeModelling/Codes/graph-bridges/results/table/table_report_0"

    table_of_results.fill_table([parent_experiment_folder,parent_experiment_folder2])

    pandas_table = table_of_results.create_pandas()
    pprint(pandas_table)

    """

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