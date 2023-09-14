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
from graph_bridges.models.metrics.sb_metrics import marginal_paths_histograms_plots, graph_metrics_for_sb
from graph_bridges.utils.plots.graph_plots import plot_graphs_list2

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
                 config:SBConfig=None):
        """
        :param paths_dataloader: contains a data distribution and a target distribution (also possibly data)
        :param backward_estimator:
        :param current_model: model to be trained
        :param past_model: model as obtained in the previous sinkhorn iteration
        :param kwargs:

             the paths_dataloader is a part of the
        """

        # select device
        self.sb_config = config
        self.device = torch.device(self.sb_config.trainer.device)

        self.sb = SB()
        self.sb.create_new_from_config(self.sb_config,self.device)

        self.starting_sinkhorn = self.sb_config.trainer.starting_sinkhorn
        self.number_of_sinkhorn = self.sb_config.trainer.number_of_sinkhorn
        self.number_of_epochs = self.sb_config.trainer.num_epochs

        # METRICS
        self.metrics = None
        self.metrics_kwargs = None

    def parameters_info(self, sinkhorn_iteration=0):
        print("# ==================================================")
        print("# START OF BACKWARD RATIO TRAINING SINKHORN {0}".format(sinkhorn_iteration))
        print("# ==================================================")
        if sinkhorn_iteration == 0:
            print("# Current Model ************************************")
            pprint(self.sb_config.model.__dict__)
            print("# Reference Parameters **********************************")
            pprint(self.sb_config.reference.__dict__)
            print("# Trainer Parameters")
            pprint(self.sb_config.trainer.__dict__)

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
        spins_path, times = self.sb.pipeline(None, 0, self.device,return_path=True,return_path_shape=True)
        batch_size, total_number_of_steps, number_of_spins = spins_path.shape[0],spins_path.shape[1],spins_path.shape[2]
        spins_path, times = self.sb.pipeline(None, 0, self.device, return_path=True)

        #CHECK LOSS
        initial_loss = self.sb.backward_ratio_stein_estimator.estimator(current_model,
                                                                        past_to_train_model,
                                                                        spins_path,
                                                                        times)
        assert torch.isnan(initial_loss).any() == False
        assert torch.isinf(initial_loss).any() == False

        #DEFINE OPTIMIZERS
        self.optimizer = Adam(current_model.parameters(), lr=self.sb_config.trainer.learning_rate)
        # histogram_path_plot_path = self.config.experiment_files.plot_path.format("initial_plot")

        # METRICS
        self.log_metrics(current_model=current_model,
                         past_to_train_model=past_to_train_model,
                         sinkhorn_iteration=sinkhorn_iteration,
                         epoch=0,
                         device=self.device)
        # INFO
        #self.parameters_info(sinkhorn_iteration)

        return initial_loss

    def train_step(self, current_model, past_model, databatch, number_of_training_step, sinkhorn_iteration=0):
        databatch = self.preprocess_data(databatch)
        # LOSS UPDATE
        loss = self.sb.backward_ratio_stein_estimator.estimator(current_model, past_model, databatch[0], databatch[1])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # SUMMARIES
        self.writer.add_scalar('training loss sinkhorn {0}'.format(sinkhorn_iteration), loss, number_of_training_step)
        return loss

    def test_step(self, current_model, past_model, databatch, number_of_test_step, sinkhorn_iteration=0):
        with torch.no_grad():
            databatch = self.preprocess_data(databatch)
            loss = self.sb.backward_ratio_stein_estimator.estimator(current_model, past_model, databatch[0], databatch[1])
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
        self.sb.backward_ratio_stein_estimator.set_device(self.device)

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
            for epoch in tqdm(range(self.number_of_epochs)):
                #TRAINING ----------------------------------------------------------------------------------------------
                training_loss = []
                for databatch in self.sb.pipeline.paths_iterator(past_model,
                                                                 sinkhorn_iteration=sinkhorn_iteration,
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
                    self.save_results(current_model=training_model,
                                      past_model=past_model,
                                      initial_loss=initial_loss,
                                      training_loss_average=training_loss_average,
                                      validation_loss_average=0.,
                                      LOSS=LOSS,
                                      epoch=epoch+1,
                                      sinkhorn_iteration=sinkhorn_iteration,
                                      checkpoint=False)

                if epoch % 10 == 0:
                    print("Epoch: {}, Loss: {}".format(epoch + 1, training_loss_average))

                if (epoch + 1) % self.sb_config.trainer.save_model_epochs == 0:
                    self.save_results(current_model=training_model,
                                      past_model=past_model,
                                      initial_loss=initial_loss,
                                      training_loss_average=training_loss_average,
                                      validation_loss_average=0.,
                                      LOSS=LOSS,
                                      epoch=epoch+1,
                                      sinkhorn_iteration=sinkhorn_iteration,
                                      checkpoint=True)

                if (epoch + 1) % self.sb_config.trainer.save_metric_epochs == 0:
                    self.log_metrics(current_model=training_model,
                                     past_to_train_model=past_model,
                                     sinkhorn_iteration=sinkhorn_iteration,
                                     epoch=epoch+1,
                                     device=self.device)

            self.time1 = datetime.now()
            #=====================================================
            # RESULTS FROM BEST MODEL UPDATED WITH METRICS
            #=====================================================
        self.writer.close()

        return best_loss

    def end_of_sinkhorn(self):
        """
        :return:
        """
        return None

    def log_metrics(self, current_model, past_to_train_model, sinkhorn_iteration, epoch, device):
        """
        After the training procedure is done, the model is updated

        :return:
        """
        config = self.sb_config

        #HISTOGRAMS
        if "histograms" in config.trainer.metrics:
            histograms_plot_path_ = config.experiment_files.plot_path.format("sinkhorn_{0}_{1}".format(sinkhorn_iteration,
                                                                                                                  epoch))
            marginal_paths_histograms_plots(self.sb,
                                            sinkhorn_iteration,
                                            device,
                                            current_model,
                                            past_to_train_model,
                                            plot_path=histograms_plot_path_)

        #METRICS
        if "graphs" in config.trainer.metrics:
            graph_metrics_path_ = config.experiment_files.metrics_file.format("graph_sinkhorn_{0}_{1}".format(sinkhorn_iteration,
                                                                                                              epoch))
            graph_metrics =  graph_metrics_for_sb(self.sb, current_model,device)
            with open(graph_metrics_path_, "w") as f:
                json.dump(graph_metrics, f)

        #PLOTS
        if "graphs_plots" in config.trainer.metrics:
            graph_plot_path_ = config.experiment_files.graph_plot_path.format("generative_sinkhorn_{0}_{1}".format(sinkhorn_iteration,epoch))
            generated_graphs = self.sb.generate_graphs(number_of_graphs=20,
                                                       generating_model=current_model,
                                                       sinkhorn_iteration=1)
            plot_graphs_list2(generated_graphs,title="Generated 0",save_dir=graph_plot_path_)

    def save_results(self,
                     current_model,
                     past_model,
                     initial_loss,
                     training_loss_average,
                     validation_loss_average,
                     LOSS,
                     epoch,
                     sinkhorn_iteration,
                     checkpoint=True):
        RESULTS = {"current_model": current_model,
                   "past_model": past_model,
                   "initial_loss": initial_loss.item(),
                   "LOSS": LOSS,
                   "best_loss": validation_loss_average,
                   "training_loss": training_loss_average,
                   "sinkhorn_iteration":sinkhorn_iteration}
        if checkpoint:
            best_model_path_checkpoint = self.sb_config.experiment_files.best_model_path_checkpoint.format(epoch, sinkhorn_iteration)
            torch.save(RESULTS,best_model_path_checkpoint)
        else:
            torch.save(RESULTS, self.sb_config.experiment_files.best_model_path.format(sinkhorn_iteration))


if __name__=="__main__":
    # CONFIGURATIONS IMPORT
    from graph_bridges.configs.graphs.graph_config_sb import SBConfig, SBTrainerConfig
    from graph_bridges.configs.graphs.graph_config_sb import SteinSpinEstimatorConfig
    from graph_bridges.configs.graphs.graph_config_sb import ParametrizedSamplerConfig
    from graph_bridges.configs.graphs.graph_config_sb import get_sb_config_from_file

    # DATA CONFIGS
    from graph_bridges.data.graph_dataloaders_config import CommunitySmallConfig, EgoConfig
    # BACKWARD RATES CONFIGS
    from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig, \
        GaussianTargetRateImageX0PredEMAConfig

    # ===========================================
    # MODEL SET UP
    # ===========================================
    sb_config = SBConfig(delete=True,
                         experiment_name="graph",
                         experiment_type="sb",
                         experiment_indentifier=None)

    sb_config.data = EgoConfig(batch_size=10, full_adjacency=False)
    sb_config.model = BackRateMLPConfig(time_embed_dim=14, hidden_layer=200)
    sb_config.stein = SteinSpinEstimatorConfig(stein_sample_size=200,
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