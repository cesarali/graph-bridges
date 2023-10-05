import json

import torch
from pprint import pprint
from datetime import datetime
from tqdm import tqdm

import numpy as np
from graph_bridges.models.generative_models.sb import SB
from graph_bridges.configs.graphs.graph_config_sb import SBConfig
from graph_bridges.utils.plots.sb_plots import sinkhorn_plot

from pathlib import Path

from torch.optim import Adam

from typing import Optional
from graph_bridges.configs.graphs.graph_config_sb import SBConfig
from graph_bridges.models.metrics.sb_metrics import  graph_metrics_for_sb
from graph_bridges.utils.plots.graph_plots import plot_graphs_list2
from graph_bridges.models.metrics.sb_paths_metrics import states_paths_histograms_plots
#from graph_bridges.models.reference_process.glauber_reference import GlauberDynamics
from graph_bridges.models.metrics.histograms_metrics import marginals_histograms_mse

from graph_bridges.models.metrics.sb_metrics import paths_marginal_histograms
from graph_bridges.models.metrics.sb_metrics_utils import log_metrics
from graph_bridges.models.backward_rates.backward_rate_utils import load_backward_rates
def copy_models(model_to, model_from):
    model_to.load_state_dict(model_from.state_dict())

check_model_devices = lambda x: x.parameters().__next__().device

class SBTrainer:
    """
    This trainer is intended to obtain a backward process of a markov jump via
    a ratio estimator with a stein estimator
    """
    sb:SB
    sb_config:SBConfig
    name_ = "schrodinger_bridge"
    def __init__(self,
                 config:SBConfig=None,
                 experiment_name=None,
                 experiment_type=None,
                 experiment_indentifier=None,
                 new_experiment_indentifier=None,
                 sinkhorn_iteration_to_load=0,
                 checkpoint=None,
                 next_sinkhorn=False):
        """
        :param paths_dataloader: contains a data distribution and a target distribution (also possibly data)
        :param backward_estimator:
        :param current_model: model to be trained
        :param past_model: model as obtained in the previous sinkhorn iteration
        :param kwargs:

             the paths_dataloader is a part of the
        """
        if config is not None:
            # select device
            self.sb_config = config
            self.device = torch.device(self.sb_config.trainer.device)
            self.sb = SB()
            self.sb.create_new_from_config(self.sb_config,self.device)
        else:
            self.sb = SB()
            self.sb.load_from_results_folder(experiment_name=experiment_name,
                                             experiment_type=experiment_type,
                                             experiment_indentifier=experiment_indentifier,
                                             new_experiment=True,
                                             new_experiment_indentifier=new_experiment_indentifier,
                                             sinkhorn_iteration_to_load=sinkhorn_iteration_to_load,
                                             checkpoint=checkpoint)
            self.sb_config = self.sb.config
            self.device = torch.device(self.sb_config.trainer.device)

            if next_sinkhorn:
                self.sb_config.trainer.starting_sinkhorn = sinkhorn_iteration_to_load + 1
                self.end_of_sinkhorn()
            else:
                self.sb_config.trainer.starting_sinkhorn = sinkhorn_iteration_to_load

            old_number_of_sinkhorn = self.sb_config.trainer.number_of_sinkhorn
            starting_sinkhorn = self.sb_config.trainer.starting_sinkhorn
            self.number_of_sinkhorn = min(old_number_of_sinkhorn,starting_sinkhorn) + 1
            self.sb_config.trainer.number_of_sinkhorn = self.number_of_sinkhorn

        self.starting_sinkhorn = self.sb_config.trainer.starting_sinkhorn
        self.number_of_sinkhorn = self.sb_config.trainer.number_of_sinkhorn
        self.number_of_epochs = self.sb_config.trainer.num_epochs
        self.clip_max_norm = self.sb_config.trainer.clip_max_norm
        self.do_ema = self.sb_config.model.do_ema

        # METRICS
        self.metrics = None
        self.metrics_kwargs = None

    def parameters_info(self, sinkhorn_iteration=0):
        print("# ==================================================")
        print("# START OF BACKWARD RATIO TRAINING SINKHORN {0}".format(sinkhorn_iteration))
        print("# ==================================================")
        if sinkhorn_iteration == 0:
            print("Identifiers ")

            print(self.config.experiment_type)
            print(self.config.experiment_name)
            print(self.config.experiment_indentifier)

        print("# ==================================================")
        print("# Number of Epochs {0}".format(self.number_of_epochs))
        print("# ==================================================")

    def initialize_sinkhorn(self, current_model, past_to_train_model, sinkhorn_iteration=0):
        """
        Obtains initial loss to know when to save, restart the optimizer

        :param current_model:
        :param past_to_train_model:
        :param sinkhorn_iteration:
        :return:
        """
        print(current_model.parameters().__next__().device)
        if sinkhorn_iteration != 0:
            print(past_to_train_model.parameters().__next__().device)

        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(self.sb_config.experiment_files.tensorboard_path)

        # CHECK DATA
        #spins_path, times = self.sb.pipeline(None, 0, self.device,return_path=True,return_path_shape=True)
        #batch_size, total_number_of_steps, number_of_spins = spins_path.shape[0],spins_path.shape[1],spins_path.shape[2]
        spins_path, times = self.sb.pipeline(past_to_train_model, sinkhorn_iteration, self.device, return_path=True)

        #CHECK LOSS
        initial_loss = self.sb.backward_ratio_estimator(current_model,
                                                        past_to_train_model,
                                                        spins_path,
                                                        times)
        assert torch.isnan(initial_loss).any() == False
        assert torch.isinf(initial_loss).any() == False

        #DEFINE OPTIMIZERS
        self.optimizer = Adam(current_model.parameters(), lr=self.sb_config.trainer.learning_rate)
        # histogram_path_plot_path = self.config.experiment_files.plot_path.format("initial_plot")

        # METRICS
        log_metrics(sb=self.sb,
                    current_model=current_model,
                    past_to_train_model=past_to_train_model,
                    sinkhorn_iteration=sinkhorn_iteration,
                    epoch=0,
                    device=self.device)
        # INFO
        #self.parameters_info(sinkhorn_iteration)
        return initial_loss

    def train_step(self, current_model, past_model, databatch, number_of_training_step, sinkhorn_iteration=0):
        databatch = self.preprocess_data(databatch)
        X_spins = databatch[0]
        current_time = databatch[1]
        # LOSS UPDATE
        loss = self.sb.backward_ratio_estimator(current_model, past_model, X_spins, current_time,sinkhorn_iteration=sinkhorn_iteration)
        self.optimizer.zero_grad()
        loss.backward()
        if self.sb_config.trainer.clip_grad:
            torch.nn.utils.clip_grad_norm_(current_model.parameters(), max_norm=self.sb_config.trainer.clip_max_norm)
        self.optimizer.step()
        if self.do_ema:
            current_model.update_ema()
        # SUMMARIES
        self.writer.add_scalar('training loss sinkhorn {0}'.format(sinkhorn_iteration), loss, number_of_training_step)
        return loss

    def test_step(self, current_model, past_model, databatch, number_of_test_step, sinkhorn_iteration=0):
        with torch.no_grad():
            databatch = self.preprocess_data(databatch)
            X_spins = databatch[0]
            current_time = databatch[1]
            loss = self.sb.backward_ratio_estimator(current_model, past_model, X_spins, current_time)
            self.writer.add_scalar('test loss sinkhorn {0}'.format(sinkhorn_iteration), loss, number_of_test_step)
        return loss

    def preprocess_data(self, databatch):
        return [databatch[0].to(self.device),
                databatch[1].to(self.device)]

    def train_schrodinger(self,past_loss=None):
        """
        FORWARD  means sampling from p_0 (data) -> p_1 (target)

        :param training_model:
        :param past_model:
        :return:
        """
        training_model = self.sb.training_model
        past_model = self.sb.past_model

        if past_loss is not None:
            past_loss = past_loss.to(self.device)
        else:
            past_loss = past_model

        assert check_model_devices(training_model) == self.device

        self.sb.backward_ratio_estimator.set_device(self.device)

        for sinkhorn_iteration in range(self.starting_sinkhorn,self.number_of_sinkhorn):
            if sinkhorn_iteration == 0:
                past_model = self.sb.reference_process
            else:
                past_model = self.sb.past_model

            #=====================================================================================
            # SINKHORN TRAINING
            #=====================================================================================
            # INITIATE LOSS
            initial_loss = self.initialize_sinkhorn(training_model, past_model, sinkhorn_iteration)

            best_loss = initial_loss
            LOSS = []
            number_of_test_step = 0
            number_of_training_step = 0
            self.time0 = datetime.now()
            results = []
            all_metrics = {}

            for epoch in tqdm(range(self.number_of_epochs)):
                #TRAINING ----------------------------------------------------------------------------------------------
                training_loss = []
                for databatch in self.sb.pipeline.paths_iterator(past_model,
                                                                 sinkhorn_iteration=sinkhorn_iteration,
                                                                 sample_from_reference_native=self.sb_config.sampler.sample_from_reference_native,
                                                                 device=self.device,
                                                                 train=True):
                    # DATA
                    loss = self.train_step(training_model,
                                           past_model,
                                           databatch,
                                           number_of_training_step,
                                           sinkhorn_iteration)
                    training_loss.append(loss.item())
                    number_of_training_step += 1
                    LOSS.append(loss.item())
                training_loss_average = np.asarray(training_loss).mean()
                #VALIDATION---------------------------------------------------------------------------------------------
                validation_loss = []
                for databatch in self.sb.pipeline.paths_iterator(past_model,
                                                                 sinkhorn_iteration=sinkhorn_iteration,
                                                                 device=self.device,
                                                                 sample_from_reference_native=self.sb_config.sampler.sample_from_reference_native,
                                                                 train=True):
                    loss = self.test_step(training_model,
                                          past_model,
                                          databatch,
                                          number_of_training_step,
                                          sinkhorn_iteration)
                    validation_loss.append(loss.item())
                    number_of_test_step +=1
                validation_loss_average = np.asarray(validation_loss).mean()
                # SAVE RESULTS IF LOSS DECREASES IN VALIDATION ---------------------------------------------------------
                if validation_loss_average < best_loss:
                    best_loss = validation_loss_average
                    results = self.save_results(current_model=training_model,
                                                past_model=past_model,
                                                initial_loss=initial_loss,
                                                training_loss_average=training_loss_average,
                                                validation_loss_average=validation_loss_average,
                                                best_loss=best_loss,
                                                LOSS=LOSS,
                                                epoch=epoch+1,
                                                sinkhorn_iteration=sinkhorn_iteration,
                                                checkpoint=False)

                if epoch % 10 == 0:
                    print("Epoch: {}, Loss: {}".format(epoch + 1, training_loss_average))

                if (epoch + 1) % self.sb_config.trainer.save_model_epochs == 0:
                    results = self.save_results(current_model=training_model,
                                                past_model=past_model,
                                                initial_loss=initial_loss,
                                                training_loss_average=training_loss_average,
                                                validation_loss_average=validation_loss_average,
                                                best_loss=best_loss,
                                                LOSS=LOSS,
                                                epoch=epoch+1,
                                                sinkhorn_iteration=sinkhorn_iteration,
                                                checkpoint=True)

                if (epoch + 1) % self.sb_config.trainer.save_metric_epochs == 0:
                    all_metrics = log_metrics(sb=self.sb,
                                              current_model=training_model,
                                              past_to_train_model=past_model,
                                              sinkhorn_iteration=sinkhorn_iteration,
                                              epoch=epoch+1,
                                              device=self.device)

            self.time1 = datetime.now()
            #=====================================================
            # RESULTS FROM BEST MODEL UPDATED WITH METRICS
            #=====================================================
            self.end_of_sinkhorn()

        self.writer.close()

        return results,all_metrics

    def end_of_sinkhorn(self):
        """
        :return:
        """
        self.sb.past_model.load_state_dict(self.sb.training_model.state_dict())
        self.sb.training_model = load_backward_rates(self.sb_config, self.device)

    def save_results(self,
                     current_model,
                     past_model,
                     initial_loss,
                     validation_loss_average,
                     training_loss_average,
                     best_loss,
                     LOSS,
                     epoch,
                     sinkhorn_iteration,
                     checkpoint=True):

        RESULTS = {"current_model": current_model,
                   "past_model": past_model,
                   "initial_loss": initial_loss.item(),
                   "LOSS": LOSS,
                   "best_loss": best_loss,
                   "training_loss": training_loss_average,
                   "validation_loss_average":validation_loss_average,
                   "sinkhorn_iteration":sinkhorn_iteration}
        if checkpoint:
            best_model_path_checkpoint = self.sb_config.experiment_files.best_model_path_checkpoint.format(epoch, sinkhorn_iteration)
            torch.save(RESULTS,best_model_path_checkpoint)
        else:
            torch.save(RESULTS, self.sb_config.experiment_files.best_model_path.format(sinkhorn_iteration))

        return RESULTS


if __name__=="__main__":
    # CONFIGURATIONS IMPORT
    from graph_bridges.configs.graphs.graph_config_sb import SBConfig, SBTrainerConfig
    from graph_bridges.configs.graphs.graph_config_sb import SteinSpinEstimatorConfig
    from graph_bridges.configs.graphs.graph_config_sb import ParametrizedSamplerConfig
    from graph_bridges.configs.graphs.graph_config_sb import get_sb_config_from_file

    # DATA CONFIGS
    from graph_bridges.data.graph_dataloaders_config import CommunitySmallConfig, EgoConfig
    # BACKWARD RATES CONFIGS
    from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig, GaussianTargetRateImageX0PredEMAConfig
    # ===========================================
    # MODEL SET UP
    # ===========================================
    sb_config = SBConfig(delete=True,
                         experiment_name="graph",
                         experiment_type="sb",
                         experiment_indentifier=None)

    sb_config.data = EgoConfig(batch_size=10, full_adjacency=False)
    sb_config.model = BackRateMLPConfig(time_embed_dim=14, hidden_layer=200)
    sb_config.flip_estimator = SteinSpinEstimatorConfig(stein_sample_size=200,
                                                        stein_epsilon=0.23)
    sb_config.sampler = ParametrizedSamplerConfig(num_steps=10)
    sb_config.trainer = SBTrainerConfig(learning_rate=1e-2,
                                        num_epochs=3000,
                                        save_metric_epochs=300,
                                        save_model_epochs=300,
                                        save_image_epochs=300,
                                        device="cuda:0",
                                        metrics=["graphs_plots",
                                                 "histograms"])

    # ========================================
    # TRAIN
    # ========================================
    sb_trainer = SBTrainer(sb_config)
    sb_trainer.train_schrodinger()