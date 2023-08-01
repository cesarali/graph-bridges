import json
import torch
import numpy as np
from pathlib import Path
from torch.optim import Adam

from pprint import pprint
from graph_bridges.models.generative_models.ctdd import CTDD
from graph_bridges.configs.graphs.config_ctdd import CTDDConfig
from graph_bridges.models.metrics.ctdd_metrics import graph_metrics_for_ctdd
from graph_bridges.models.metrics.ctdd_metrics import marginal_histograms_for_ctdd

from graph_bridges.utils.plots.histograms_plots import plot_histograms
from graph_bridges.utils.plots.graph_plots import plot_graphs_list2

class Standard():
    def __init__(self, cfg):
        self.do_ema = 'ema_decay' in cfg.model
        self.clip_grad = cfg.training.clip_grad
        self.warmup = cfg.training.warmup
        self.lr = cfg.optimizer.lr

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
    name_ = "continuos_time_discrete_denoising"

    def __init__(self,
                 ctdd:CTDD,
                 **kwargs):
        """
        :param paths_dataloader: contains a data distribution and a target distribution (also possibly data)
        :param backward_estimator:
        :param current_model: model to be trained
        :param past_model: model as obtained in the previous sinkhorn iteration
        :param kwargs:

             the paths_dataloader is a part of the
        """
        self.ctdd =  ctdd
        self.config = self.ctdd.config
        self.number_of_epochs = self.config.optimizer.num_epochs

        # select device
        self.cuda = kwargs.get("cuda")
        if self.cuda is not None:
            self.device = torch.device('cuda:{0}'.format(self.cuda) if torch.cuda.is_available() else "cuda")
        else:
            self.device = torch.device("cpu")

    def parameters_info(self, sinkhorn_iteration=0):
        print("# ==================================================")
        print("# START OF BACKWARD RATIO TRAINING CTDD".format(sinkhorn_iteration))
        print("# ==================================================")

        print("# Current Model ************************************")
        pprint(self.config.model.__dict__)
        print("# Reference Parameters **********************************")
        pprint(self.config.reference.__dict__)
        print("# Trainer Parameters")
        pprint(self.config.optimizer.__dict__)

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
        self.optimizer = Adam(self.ctdd.model.parameters(), lr=self.config.optimizer.learning_rate)

        # CHECK DATA
        databatch = next(self.ctdd.data_dataloader.train().__iter__())
        x_adj = databatch[0]
        x_features =  databatch[1]
        print(x_adj.shape)
        print(x_features.shape)

        #CHECK LOSS
        initial_loss = self.train_step(self.ctdd.model, databatch, 0)
        assert torch.isnan(initial_loss).any() == False
        assert torch.isinf(initial_loss).any() == False

        # METRICS
        self.log_metrics(0,self.device)

        #SAVE INITIAL STUFF

        return initial_loss

    def train_step(self, current_model, databatch, number_of_training_step):
        databatch = self.preprocess_data(databatch)
        x_adj, x_features = databatch[0],databatch[1]
        B = x_adj.shape[0]

        # Sample a random timestep for each image
        ts = torch.rand((B,), device=device) * (1.0 - config.loss.min_time) + config.loss.min_time

        x_t, x_tilde, qt0, rate = self.ctdd.scheduler.add_noise(x_adj, self.ctdd.reference_process, ts, device, return_dict=False)
        x_logits, p0t_reg, p0t_sig, reg_x = current_model(x_adj, ts, x_tilde)

        self.optimizer.zero_grad()
        loss_ = self.ctdd.loss(x_adj, x_tilde, qt0, rate, x_logits, reg_x, p0t_sig, p0t_reg, device)
        loss_.backward()
        self.optimizer.step()

        # SUMMARIES
        self.writer.add_scalar('training loss', loss_.item(), number_of_training_step)
        return loss_

    def test_step(self, current_model, databatch, number_of_test_step, sinkhorn_iteration=0):
        with torch.no_grad():
            databatch = self.preprocess_data(databatch)
            loss = None
            self.writer.add_scalar('test loss sinkhorn {0}'.format(sinkhorn_iteration), loss, number_of_test_step)
        return loss

    def preprocess_data(self, databatch):
        return [databatch[0].to(self.device),
                databatch[1].to(self.device)]

    def train_ctdd(self,past_loss=None):
        """
        FORWARD  means sampling from p_0 (data) -> p_1 (target)

        :param training_model:
        :param past_model:
        :return:
        """
        # SET GPUS
        training_model = self.ctdd.model
        training_model = training_model.to(self.device)

        # INITIATE LOSS
        initial_loss = self.initialize(training_model)
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

    def log_metrics(self,number_of_steps,device):
        """
        After the training procedure is done, the model is updated

        :return:
        """
        config = self.config

        #HISTOGRAMS
        histograms_plot_path_ = config.experiment_files.plot_path.format("histograms_{0}".format(number_of_steps))
        marginal_histograms = marginal_histograms_for_ctdd(self.ctdd,config,device)
        plot_histograms(marginal_histograms, plots_path=histograms_plot_path_)

        #METRICS
        #graph_metrics_path_ = config.experiment_files.metrics_file.format("graph_{0}".format(number_of_steps))
        #graph_metrics = graph_metrics_for_ctdd(self.ctdd, config)
        #with open(graph_metrics_path_, "w") as f:
        #    json.dump(graph_metrics, f)

        #PLOTS
        graph_plot_path_ = config.experiment_files.graph_plot_path.format("generative_{0}".format(number_of_steps))
        generated_graphs = self.ctdd.generate_graphs(number_of_graphs=36)
        plot_graphs_list2(generated_graphs,title="Generated 0",save_dir=graph_plot_path_)

    def save_results(self,
                     current_model,
                     initial_loss,
                     training_loss_average,
                     validation_loss_average, LOSS):
        RESULTS = {"current_model": current_model,
                   "initial_loss": initial_loss.item(),
                   "LOSS": LOSS,
                   "best_loss": validation_loss_average,
                   "training_loss": training_loss_average}
        #torch.save(RESULTS,
        #           self.best_model_path.format(sinkhorn_iteration))


if __name__=="__main__":
    # CONFIGS
    from graph_bridges.configs.graphs.config_ctdd import CTDDConfig
    from graph_bridges.data.graph_dataloaders_config import EgoConfig
    from graph_bridges.models.backward_rates.backward_rate_config import GaussianTargetRateImageX0PredEMAConfig
    from graph_bridges.models.generative_models.ctdd import CTDD

    #==================================================================
    # CREATE OBJECTS FROM CONFIGURATION

    device = torch.device("cpu")
    config = CTDDConfig(experiment_indentifier="training_test")
    config.data = EgoConfig(as_image=False, batch_size=32, full_adjacency=False)
    config.model = GaussianTargetRateImageX0PredEMAConfig()

    ctdd = CTDD()
    ctdd.create_new_from_config(config,device)


    ctdd_trainer = CTDDTrainer(ctdd)
    ctdd_trainer.initialize()

