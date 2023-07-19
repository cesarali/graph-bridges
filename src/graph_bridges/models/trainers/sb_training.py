import json

import torch
from pprint import pprint
from datetime import datetime
from tqdm import tqdm

import numpy as np
from graph_bridges.models.generative_models.sb import SB
from graph_bridges.configs.graphs.config_sb import BridgeConfig
from graph_bridges.utils.plots.sb_plots import sinkhorn_plot

from pathlib import Path

from torch.optim import Adam

from typing import Optional


def copy_models(model_to, model_from):
    model_to.load_state_dict(model_from.state_dict())

check_model_devices = lambda x: x.parameters().__next__().device


class SBTrainer:
    """
    This trainer is intended to obtain a backward process of a markov jump via
    a ratio estimator with a stein estimator
    """
    name_ = "schrodinger_bridge"
    def __init__(self,
                 config:BridgeConfig=Optional,
                 sb:SB=None,
                 **kwargs):
        """
        :param paths_dataloader: contains a data distribution and a target distribution (also possibly data)
        :param backward_estimator:
        :param current_model: model to be trained
        :param past_model: model as obtained in the previous sinkhorn iteration
        :param kwargs:

             the paths_dataloader is a part of the
        """
        if config is not None:
            self.config = config
            sb = SB()
            sb.create_from_config(config, device)
        else:
            self.sb = sb
            self.config = self.sb.config

        self.config.initialize()
        self.starting_sinkhorn = self.config.optimizer.starting_sinkhorn
        self.number_of_sinkhorn = self.config.optimizer.number_of_sinkhorn
        self.number_of_epochs = self.config.optimizer.num_epochs

        # METRICS
        self.metrics = None
        self.metrics_kwargs = None

        # select device
        self.cuda = kwargs.get("cuda")
        if self.cuda is not None:
            self.device = torch.device('cuda:{0}'.format(self.cuda) if torch.cuda.is_available() else "cuda")
        else:
            self.device = torch.device("cpu")

    def parameters_info(self, sinkhorn_iteration=0):
        print("# ==================================================")
        print("# START OF BACKWARD RATIO TRAINING SINKHORN {0}".format(sinkhorn_iteration))
        print("# ==================================================")
        if sinkhorn_iteration == 0:
            print("# Current Model ************************************")
            pprint(self.config.model.__dict__)
            print("# Reference Parameters **********************************")
            pprint(self.config.reference.__dict__)
            print("# Trainer Parameters")
            pprint(self.config.optimizer.__dict__)

        print("# ==================================================")
        print("# Number of Epochs {0}".format(self.number_of_epochs))
        print("# ==================================================")

    def initialize(self, current_model, past_to_train_model, device, sinkhorn_iteration=0):
        """
        Obtains initial loss to know when to save, restart the optimizer

        :param current_model:
        :param past_to_train_model:
        :param sinkhorn_iteration:
        :return:
        """
        if sinkhorn_iteration != 0:
            print(past_to_train_model.parameters().__next__().device)

        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(self.config.tensorboard_path)

        # CHECK DATA
        spins_path, times = self.sb.pipeline(None, 0, device,return_path=True,return_path_shape=True)
        batch_size, total_number_of_steps, number_of_spins = spins_path.shape[0],spins_path.shape[1],spins_path.shape[2]

        spins_path, times = self.sb.pipeline(None, 0, device, return_path=True)
        #CHECK LOSS
        initial_loss = self.sb.backward_ration_stein_estimator.estimator(current_model,
                                                                  past_to_train_model,
                                                                  spins_path,
                                                                  times)
        assert torch.isnan(initial_loss).any() == False
        assert torch.isinf(initial_loss).any() == False

        #DEFINE OPTIMIZERS
        self.optimizer = Adam(current_model.parameters(), lr=self.config.optimizer.learning_rate)

        #CHECK DATA METRICS
        data_stats_path = Path(self.sb.config.data_stats)
        if data_stats_path.exists():
            data_stats = json.load(open(data_stats_path,"rb"))
        else:
            from graph_bridges.models.metrics.data_metrics import SpinBernoulliMarginal

            bernoulli_marginal = SpinBernoulliMarginal(spin_dataloader=self.sb.data_dataloader)
            marginal_0 = bernoulli_marginal()
            bernoulli_marginal = SpinBernoulliMarginal(spin_dataloader=self.sb.target_dataloader)
            marginal_1 = bernoulli_marginal()

            histogram_path_0 = torch.zeros(total_number_of_steps,number_of_spins)
            histogram_path_1 = torch.zeros(total_number_of_steps,number_of_spins)

            for spins_path_1, times_1 in self.sb.pipeline.paths_iterator(past_to_train_model, sinkhorn_iteration=sinkhorn_iteration,return_path_shape=True):
                #end_of_path = spins_path_1[:, -1, :].unsqueeze(1).unsqueeze(1)
                end_of_path = spins_path_1[:, -1, :]
                spins_path_2, times_2 = self.sb.pipeline(current_model,
                                                         sinkhorn_iteration + 1,
                                                         device,
                                                         end_of_path,
                                                         return_path=True,
                                                         return_path_shape=True)
                binary_0 = (spins_path_1 + 1.)*.5
                binary_1 = (spins_path_2 + 1.)*.5

                current_sum_0 = binary_0.sum(axis=0)
                current_sum_1 = binary_1.sum(axis=0)

                histogram_path_0 += current_sum_0
                histogram_path_1 += current_sum_1

            print(histogram_path_0[0])
            print(histogram_path_0[-1])

            print(histogram_path_1[0])
            print(histogram_path_1[-1])

            state_legends = [str(i) for i in range(histogram_path_0.shape[-1])]
            sinkhorn_plot(sinkhorn_iteration,
                          marginal_0,
                          marginal_1,
                          backward_histogram=histogram_path_1,
                          forward_histogram=histogram_path_0,
                          time_=times_1,
                          states_legends=state_legends)

        # INFO
        #self.parameters_info(sinkhorn_iteration)

        return 0

    def train_step(self, current_model, past_model, databatch, number_of_training_step, sinkhorn_iteration=0):
        databatch = self.preprocess_data(databatch)
        # LOSS UPDATE
        loss = self.backward_estimator.estimator(current_model, past_model, databatch[0], databatch[1])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # SUMMARIES
        self.writer.add_scalar('training loss sinkhorn {0}'.format(sinkhorn_iteration), loss, number_of_training_step)
        return loss

    def test_step(self, current_model, past_model, databatch, number_of_test_step, sinkhorn_iteration=0):
        with torch.no_grad():
            databatch = self.preprocess_data(databatch)
            loss = self.backward_estimator.estimator(current_model, past_model, databatch[0], databatch[1])
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
        # SET GPUS
        training_model = self.sb.training_model
        past_model = self.sb.past_model

        if past_loss is not None:
            past_loss = past_loss.to(self.device)
        else:
            past_loss = past_model

        training_model = training_model.to(self.device)
        self.sb.pipeline.bridge_config.sampler.num_steps = 20
        self.sb.backward_ration_stein_estimator.set_device(self.device)

        for sinkhorn_iteration in range(self.starting_sinkhorn,self.number_of_sinkhorn):
            if sinkhorn_iteration == 0:
                past_model = self.sb.reference_process
            else:
                past_model = self.sb.past_model

            # INITIATE LOSS
            initial_loss = self.initialize(training_model, past_model, sinkhorn_iteration)
            best_loss = initial_loss

            """
            LOSS = []
            number_of_test_step = 0
            number_of_training_step = 0
            self.time0 = datetime.now()
            for epoch in tqdm(range(self.number_of_epochs)):
                #TRAINING
                training_loss = []
                for spins_path, times in sb.pipeline.paths_iterator(past_model,
                                                                    sinkhorn_iteration=sinkhorn_iteration,
                                                                    device=self.device,
                                                                    train=True):

                    loss = self.sb.backward_ration_stein_estimator.estimator(training_model,
                                                                             past_loss,
                                                                             spins_path,
                                                                             times)
                    # DATA
                    #loss = self.train_step(current_model,
                    #                       past_to_train,
                    #                       databatch,
                    #                       number_of_training_step,
                    #                       sinkhorn_iteration)

                    training_loss.append(loss.item())
                    number_of_training_step += 1
                    LOSS.append(loss.item())

                training_loss_average = np.asarray(training_loss).mean()

                #VALIDATION
                validation_loss = []
                for spins_path, times in sb.pipeline.paths_iterator(past_model,
                                                                    sinkhorn_iteration=sinkhorn_iteration,
                                                                    device=self.device,
                                                                    train=False):
                    loss = self.sb.backward_ration_stein_estimator.estimator(training_model,
                                                                             past_loss,
                                                                             spins_path,
                                                                             times)
                    validation_loss.append(loss.item())
                    number_of_test_step +=1
                validation_loss_average = np.asarray(validation_loss).mean()

                # SAVE RESULTS IF LOSS DECREASES IN VALIDATION
                if validation_loss_average < best_loss:
                    self.save_results(training_model,
                                      past_model,
                                      initial_loss,
                                      training_loss_average,
                                      validation_loss_average,
                                      LOSS,
                                      sinkhorn_iteration)

                if epoch % 10 == 0:
                    print("Epoch: {}, Loss: {}".format(epoch + 1, training_loss_average))

            self.time1 = datetime.now()
            #=====================================================
            # RESULTS FROM BEST MODEL UPDATED WITH METRICS
            #=====================================================

        self.writer.close()
        """
        return best_loss

    def update_final_results_with_metrics(self,
                                          RESULTS,
                                          sinkhorn_iteration,
                                          save_path_plot=None):
        """
        After the training procedure is done, the model is updated

        :return:
        """
        current_model = RESULTS["current_model"].to(device)
        past_model = RESULTS["past_model"].to(device)

        for metric_string in self.metrics:
            metric_kwargs = self.metrics_kwargs[metric_string]
            metric_kwargs.update({"sinkhorn_iteration":sinkhorn_iteration,
                                  "save_path_plot":save_path_plot})
            #metric_value = all_metrics[metric_string](self.paths_dataloader,
            #                                          current_model,
            #                                          past_model,
            #                                          **metric_kwargs)
            metric_value = {}
            if isinstance(metric_value,dict):
                RESULTS = RESULTS | metric_value
            RESULTS.update({metric_string:metric_value})


        training_time = (self.time1 - self.time0).total_seconds()
        RESULTS["training_time"] = training_time
        torch.save(RESULTS,
                   self.best_model_path.format(sinkhorn_iteration))

    def save_results(self, current_model, past_model,initial_loss,
                     training_loss_average, validation_loss_average, LOSS, sinkhorn_iteration):
        RESULTS = {"current_model": current_model,
                   "past_model": past_model,
                   "initial_loss": initial_loss.item(),
                   "LOSS": LOSS,
                   "best_loss": validation_loss_average,
                   "training_loss": training_loss_average,
                   "sinkhorn_iteration":sinkhorn_iteration}
        torch.save(RESULTS,
                   self.best_model_path.format(sinkhorn_iteration))


if __name__=="__main__":
    from graph_bridges.configs.graphs.config_sb import BridgeConfig, get_config_from_file
    from graph_bridges.data.dataloaders_config import GraphSpinsDataLoaderConfig
    from dataclasses import asdict

    config = BridgeConfig(experiment_indentifier=None)
    config.data = GraphSpinsDataLoaderConfig()

    #read the model
    #config = get_config_from_file("graph", "lobster", "1687884918")
    device = torch.device("cpu")
    sb = SB(config, device)
    #pprint(sb.config.__dict__)
    #print(config.config_path)
    #json.dump(asdict(config),config.config_path)


    sb_trainer = SBTrainer(None,sb)
    sb_trainer.train_schrodinger()

