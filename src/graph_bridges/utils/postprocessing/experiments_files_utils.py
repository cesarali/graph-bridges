import os
import json
import torch
import sqlite3
import numpy as np
import pandas as pd
from pprint import pprint

from matplotlib import pyplot as plt
from discrete_diffusion.data.paths_dataloaders import DataloaderFactory, PathsDataloader
from discrete_diffusion.models.estimators.stein_ising import SteinSpinEstimator, BackwardRatioEstimator
from discrete_diffusion.models.estimators.one_step import BackwardOneStepEstimator

from discrete_diffusion.models.estimators.backward_trainer import BackwardRatioTrainer

def obtain_models_from_results_old(RESULTS):
    """
    params
    ------
        RESULTS: dict as obtained from a BackwardRatioTrained class

    results
    -------
        backward_model,forward_model,paths_dataloader
    """
    # models
    backward_model = RESULTS["model_backward"]
    forward_model = RESULTS["model_trained"]

    # data loaders
    spins_0_parameters = RESULTS["dataloader_parameters"]["spins_0"]
    spins_1_parameters = RESULTS["dataloader_parameters"]["spins_1"]
    spins_0_name = spins_0_parameters.get("name")
    spins_1_name = spins_1_parameters.get("name")
    spins_0_parameters.update({"remove": False})
    spins_1_parameters.update({"remove": False})

    dataloader_factory = DataloaderFactory()
    spins_dataloader_0 = dataloader_factory.create(spins_0_name, **spins_0_parameters)
    spins_dataloader_1 = dataloader_factory.create(spins_1_name, **spins_1_parameters)

    paths_dataloader = PathsDataloader(spins_dataloader_0,
                                       spins_dataloader_1,
                                       **RESULTS["dataloader_parameters"])

    return backward_model, forward_model, paths_dataloader

def obtain_models_from_results(RESULTS,sinkhorn_iteration=0,remove_dataloaders=False):
    """
    params
    ------
        RESULTS: dict as obtained from a BackwardRatioTrained class

    results
    -------
        backward_model,forward_model,paths_dataloader
    """

    device = torch.device("cpu")

    # data loaders
    spins_0_parameters = RESULTS["dataloader_parameters"]["spins_0"]
    spins_1_parameters = RESULTS["dataloader_parameters"]["spins_1"]
    spins_0_name = spins_0_parameters.get("name")
    spins_1_name = spins_1_parameters.get("name")

    spins_0_parameters.update({"remove": False})
    spins_1_parameters.update({"remove": False})

    dataloader_factory = DataloaderFactory()
    spins_dataloader_0 = dataloader_factory.create(spins_0_name, **spins_0_parameters)
    spins_dataloader_1 = dataloader_factory.create(spins_1_name, **spins_1_parameters)

    paths_dataloader = PathsDataloader(spins_dataloader_0,
                                       spins_dataloader_1,
                                       **RESULTS["dataloader_parameters"])

    # models
    if sinkhorn_iteration != 0:
        past_model = RESULTS["past_model"]
        past_model.to(device)
    else:
        past_model = paths_dataloader.reference_process
    current_model = RESULTS["current_model"]
    current_model.to(device)

    # estimators
    backward_estimator_parameters = RESULTS["backward_estimator_parameters"]

    if backward_estimator_parameters["name"] == "backward_one_shot_estimator":
        ratio_estimator = BackwardOneStepEstimator(**backward_estimator_parameters)
    else:
        ratio_estimator = BackwardRatioEstimator(**backward_estimator_parameters)

    ratio_estimator.set_device(device)

    #trainers
    trainer_parameters = RESULTS["trainer_parameters"]
    ratio_trainer = BackwardRatioTrainer(paths_dataloader,
                                         ratio_estimator,
                                         current_model,
                                         past_model,
                                         **trainer_parameters)

    sinkhorn_iteration = RESULTS["sinkhorn_iteration"]

    return past_model, current_model, ratio_trainer, paths_dataloader,sinkhorn_iteration


def get_results_dictionary(experiment_number,
                           model_name="ratio_estimator",
                           experiment_type="dimension",
                           sinkhorn_iteration=0):
    from discrete_diffusion import results_path as results_dir

    all_experiments_dir = os.path.join(results_dir, model_name)
    experiment_dir = os.path.join(all_experiments_dir, "{0}_{1}".format(experiment_type,
                                                                        experiment_number))

    try:
        best_results_path = os.path.join(experiment_dir, "sinkhorn_{0}.tr".format(sinkhorn_iteration))
        RESULTS = torch.load(best_results_path)
    except:
        best_results_path = os.path.join(experiment_dir, "best_model.tr")
        RESULTS = torch.load(best_results_path)
    return RESULTS


def check_for_file(experiment_dir,file_to_check='sinkhorn_0.tr'):
    return file_to_check in os.listdir(experiment_dir)

def obtain_string_in_experiments(model_name,
                                 sinkhorn_iteration = 0,
                                 strings_in_results =["training_time"]):
    """
    Checks in a results folder file for RESULTS with a given string

    :param model_name:
    :param sinkhorn_iteration:
    :param strings_in_results:
    :return:
    """
    from discrete_diffusion import results_path
    all_experiments_dir = os.path.join(results_path,model_name)
    all_experiments = os.listdir(all_experiments_dir)
    file_to_result = {}

    for experiment_dir_local in all_experiments:
        experiment_dir = os.path.join(all_experiments_dir,experiment_dir_local)
        if check_for_file(experiment_dir,'sinkhorn_{0}.tr'.format(sinkhorn_iteration)):
            best_results_path = os.path.join(experiment_dir, "sinkhorn_{0}.tr".format(sinkhorn_iteration))
            RESULTS = torch.load(best_results_path)
            results_from_strings = {}
            for string_ in strings_in_results:
                try:
                    results_ = RESULTS.get(string_)
                    results_from_strings[string_] = results_
                except:
                    results_from_strings[string_] =  None
        else:
            results_from_strings = False
        file_to_result[experiment_dir_local] = results_from_strings
    return file_to_result

if __name__=="__main__":
    from discrete_diffusion.models.estimators.generative_model import GraphDiffusionModel

    # FROM ONE RESULT----------------------------
    diffusion_model_1 = GraphDiffusionModel(model_identifier=1683568825,
                                            model_name="graphs",
                                            experiment_type="lobster_to_efficient",
                                            sinkhorn_iteration=0)

    pprint(diffusion_model_1.RESULTS.keys())
    # OBTAIN RESULTS OF EXPERIMENTS ------------------------------------------------------------

    string_in_results = "training_time"
    model_name = "graphs"
    experiment_type = "lobster_to_efficient"

    string_experiment = obtain_string_in_experiments(model_name,
                                                     sinkhorn_iteration=0,
                                                     strings_in_results=["past_model",
                                                                         "training_time"])

    pprint(string_experiment)
    #pprint(paths_dataloader.obtain_parameters())
