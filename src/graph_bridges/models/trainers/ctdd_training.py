import json
import torch
import numpy as np
from pathlib import Path
from torch.optim import Adam

from pprint import pprint
from graph_bridges.models.generative_models.ctdd import CTDD
from graph_bridges.configs.graphs.graph_config_ctdd import CTDDConfig
from graph_bridges.models.metrics.ctdd_metrics import graph_metrics_for_ctdd
from graph_bridges.models.metrics.ctdd_metrics import marginal_histograms_for_ctdd

from graph_bridges.utils.plots.histograms_plots import plot_histograms
from graph_bridges.utils.plots.graph_plots import plot_graphs_list2
from datetime import datetime
from tqdm import tqdm

class Standard():
    def __init__(self, cfg):
        self.do_ema = 'ema_decay' in cfg.model
        self.clip_grad = cfg.training.clip_grad
        self.warmup = cfg.training.warmup
        self.lr = cfg.trainer.lr

    def step(self, state, minibatch, loss, writer):
        state['optimizer'].zero_grad()
        l = loss.calc_loss(minibatch, state, writer)

        if l.isnan().any() or l.isinf().any():
            print("Loss is nan")
            assert False
        l.backward()

        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(state['model'].parameters(), 1.0)

        if self.warmup > 0:
            for g in state['optimizer'].param_groups:
                g['lr'] = self.lr * np.minimum(state['n_iter'] / self.warmup, 1.0)

        state['optimizer'].step()

        if self.do_ema:
            state['model'].update_ema()

        writer.add_scalar('loss', l.detach(), state['n_iter'])

        return l.detach()

class CTDDTrainer:
    """
    This trainer is intended to obtain a backward process of a markov jump via
    a ratio estimator with a stein estimator
    """

    config: CTDDConfig
    ctdd: CTDD
    name_ = "continuos_time_discrete_denoising"

    def __init__(self,
                 config:CTDDConfig,
                 **kwargs):
        """
        :param paths_dataloader: contains a data distribution and a target distribution (also possibly data)
        :param backward_estimator:
        :param current_model: model to be trained
        :param past_model: model as obtained in the previous sinkhorn iteration
        :param kwargs:

             the paths_dataloader is a part of the
        """
        self.config = config
        self.number_of_epochs = self.config.trainer.num_epochs
        device_str = self.config.trainer.device
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.ctdd = CTDD()
        self.ctdd.create_new_from_config(self.config,self.device)

    def parameters_info(self, sinkhorn_iteration=0):
        print("# ==================================================")
        print("# START OF BACKWARD RATIO TRAINING CTDD".format(sinkhorn_iteration))
        print("# ==================================================")

        print("# Current Model ************************************")
        pprint(self.config.model.__dict__)
        print("# Reference Parameters **********************************")
        pprint(self.config.reference.__dict__)
        print("# Trainer Parameters")
        pprint(self.config.trainer.__dict__)

        print("# ==================================================")
        print("# Number of Epochs {0}".format(self.number_of_epochs))
        print("# ==================================================")

    def initialize(self):
        """
        Obtains initial loss to know when to save, restart the optimizer

        :param current_model:
        :param past_to_train_model:
        :param sinkhorn_iteration:
        :return:
        """

        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(self.config.experiment_files.tensorboard_path)

        #DEFINE OPTIMIZERS
        self.optimizer = Adam(self.ctdd.model.parameters(), lr=self.config.trainer.learning_rate)

        # CHECK DATA
        databatch = next(self.ctdd.data_dataloader.train().__iter__())
        databatch = self.preprocess_data(databatch)
        #CHECK LOSS
        initial_loss = self.train_step(self.ctdd.model, databatch, 0)
        assert torch.isnan(initial_loss).any() == False
        assert torch.isinf(initial_loss).any() == False

        # METRICS
        self.log_metrics(0,self.device)

        #SAVE INITIAL STUFF
        return initial_loss

    def train_step(self, current_model, databatch, number_of_training_step):
        with torch.autograd.set_detect_anomaly(True):
            databatch = self.preprocess_data(databatch)
            x_adj, x_features = databatch[0],databatch[1]
            B = x_adj.shape[0]

            # Sample a random timestep for each image
            ts = torch.rand((B,), device=self.device) * (1.0 - self.config.loss.min_time) + self.config.loss.min_time

            x_t, x_tilde, qt0, rate = self.ctdd.scheduler.add_noise(x_adj, self.ctdd.reference_process, ts, self.device, return_dict=False)
            x_logits, p0t_reg, p0t_sig, reg_x = current_model(x_adj, ts, x_tilde)

            self.optimizer.zero_grad()
            loss_ = self.ctdd.loss(x_adj, x_tilde, qt0, rate, x_logits, reg_x, p0t_sig, p0t_reg, self.device)
            loss_.backward()
            self.optimizer.step()

            # SUMMARIES
            self.writer.add_scalar('training loss', loss_.item(), number_of_training_step)
            return loss_

    def test_step(self, current_model, databatch, number_of_test_step):
        with torch.no_grad():
            databatch = self.preprocess_data(databatch)
            x_adj, x_features = databatch[0], databatch[1]
            B = x_adj.shape[0]

            # Sample a random timestep for each image
            ts = torch.rand((B,), device=self.device) * (1.0 - self.config.loss.min_time) + self.config.loss.min_time
            x_t, x_tilde, qt0, rate = self.ctdd.scheduler.add_noise(x_adj, self.ctdd.reference_process, ts, self.device,return_dict=False)
            x_logits, p0t_reg, p0t_sig, reg_x = current_model(x_adj, ts, x_tilde)
            loss_ = self.ctdd.loss(x_adj, x_tilde, qt0, rate, x_logits, reg_x, p0t_sig, p0t_reg, self.device)
            self.writer.add_scalar('test loss', loss_, number_of_test_step)

        return loss_

    def preprocess_data(self, databatch):
        return [databatch[0].float().to(self.device),
                databatch[1].float().to(self.device)]

    def train_ctdd(self):
        """
        FORWARD  means sampling from p_0 (data) -> p_1 (target)

        :param training_model:
        :param past_model:
        :return:
        """

        training_model = self.ctdd.model
        # INITIATE LOSS
        initial_loss = self.initialize()
        best_loss = initial_loss

        LOSS = []
        number_of_test_step = 0
        number_of_training_step = 0
        self.time0 = datetime.now()
        for epoch in tqdm(range(self.number_of_epochs)):
            #TRAINING
            training_loss = []
            for step, databatch in enumerate(self.ctdd.data_dataloader.train()):
                databatch = self.preprocess_data(databatch)
                # DATA
                loss = self.train_step(training_model,
                                       databatch,
                                       number_of_training_step)

                training_loss.append(loss.item())
                number_of_training_step += 1
                LOSS.append(loss.item())

                if number_of_training_step % self.config.trainer.log_loss == 0:
                    training_loss_average = np.asarray(training_loss).mean()
                    print("Epoch: {}, Loss: {}".format(epoch + 1, training_loss_average))

            training_loss_average = np.asarray(training_loss).mean()

            #VALIDATION
            validation_loss = []
            for step, databatch in enumerate(self.ctdd.data_dataloader.test()):
                databatch = self.preprocess_data(databatch)
                # DATA
                loss = self.test_step(training_model,
                                      databatch,
                                      number_of_test_step)

                training_loss.append(loss.item())
                number_of_training_step += 1
                LOSS.append(loss.item())
                validation_loss.append(loss.item())
                number_of_test_step +=1
            validation_loss_average = np.asarray(validation_loss).mean()

            if epoch % self.config.trainer.save_model_epochs == 0:
                self.save_results(current_model=training_model,
                                  initial_loss=initial_loss,
                                  training_loss_average=training_loss_average,
                                  validation_loss_average=validation_loss_average,
                                  LOSS=LOSS,
                                  number_of_training_step=number_of_training_step,
                                  checkpoint=True)

            if (epoch + 1) % self.config.trainer.save_metric_epochs == 0:
                self.log_metrics(epoch, self.device)

            # SAVE RESULTS IF LOSS DECREASES IN VALIDATION
            if validation_loss_average < best_loss:
                self.save_results(current_model=training_model,
                                  initial_loss=initial_loss,
                                  training_loss_average=training_loss_average,
                                  validation_loss_average=validation_loss_average,
                                  LOSS=LOSS,
                                  number_of_training_step=number_of_training_step,
                                  checkpoint=False)



        self.time1 = datetime.now()
        #=====================================================
        # RESULTS FROM BEST MODEL UPDATED WITH METRICS
        #=====================================================

        self.writer.close()
        return best_loss

    def log_metrics(self,number_of_steps,device):
        """
        After the training procedure is done, the model is updated

        :return:
        """
        config = self.config

        #HISTOGRAMS
        if "histograms" in config.trainer.metrics:
            histograms_plot_path_ = config.experiment_files.plot_path.format("histograms_{0}".format(number_of_steps))
            marginal_histograms = marginal_histograms_for_ctdd(self.ctdd,config,device)
            plot_histograms(marginal_histograms, plots_path=histograms_plot_path_)

        #METRICS
        if "graphs" in config.trainer.metrics:
            graph_metrics_path_ = config.experiment_files.metrics_file.format("graph_{0}".format(number_of_steps))
            graph_metrics = graph_metrics_for_ctdd(self.ctdd, device)
            with open(graph_metrics_path_, "w") as f:
                json.dump(graph_metrics, f)

        #PLOTS
        if "graph_plot" in config.trainer.metrics:
            graph_plot_path_ = config.experiment_files.graph_plot_path.format("generative_{0}".format(number_of_steps))
            generated_graphs = self.ctdd.generate_graphs(number_of_graphs=36)
            plot_graphs_list2(generated_graphs,title="Generated 0",save_dir=graph_plot_path_)

    def save_results(self,
                     current_model,
                     initial_loss,
                     training_loss_average,
                     validation_loss_average,
                     LOSS,
                     number_of_training_step,
                     checkpoint=True):
        RESULTS = {"current_model": current_model,
                   "initial_loss": initial_loss.item(),
                   "LOSS": LOSS,
                   "best_loss": validation_loss_average,
                   "training_loss": training_loss_average}
        if checkpoint:
            best_model_path_checkpoint = self.config.experiment_files.best_model_path_checkpoint.format(number_of_training_step)
            torch.save(RESULTS,best_model_path_checkpoint)
        else:
            torch.save(RESULTS, self.config.experiment_files.best_model_path)


