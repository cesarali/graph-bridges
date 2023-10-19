import torch
from torch.optim import Adam

import numpy as np
from typing import List
from pprint import pprint
from dataclasses import asdict,dataclass

from graph_bridges.configs.config_oops import OopsConfig
from graph_bridges.data.image_dataloaders import NISTLoader
from graph_bridges.models.generative_models.oops import OOPS


@dataclass
class ContrastiveDivergenceTrainerState:
    oops: OOPS
    initial_loss : float
    average_train_loss : float
    average_test_loss : float
    LOSS: List[float]

class ContrastiveDivergenceTrainer:
    """

    """
    name_="contrastive_divergence"

    def __init__(self,
                 config: OopsConfig,
                 dataloader:NISTLoader=None,
                 oops:OOPS=None):

        #set parameter values
        self.config = config
        self.learning_rate = config.trainer.learning_rate
        self.number_of_epochs = config.trainer.number_of_epochs
        self.device = torch.device(config.trainer.device)

        #define models
        self.oops = OOPS()
        self.oops.create_new_from_config(self.config, self.device)
        self.dataloader = self.oops.dataloader

    def parameters_info(self):
        print("# ==================================================")
        print("# START OF BACKWARD MI TRAINING ")
        print("# ==================================================")
        print("# VAE parameters ************************************")
        pprint(asdict(self.oops.config))
        print("# Paths Parameters **********************************")
        pprint(asdict(self.dataloader.config))
        print("# Trainer Parameters")
        pprint(asdict(self.config))
        print("# ==================================================")
        print("# Number of Epochs {0}".format(self.number_of_epochs))
        print("# ==================================================")

    def preprocess_data(self,data_batch):
        if len(data_batch) > 1:
            return (data_batch[0].to(self.device), data_batch[1].to(self.device))
        else:
            return (data_batch[0].to(self.device),)

    def train_step(self,data_batch,number_of_training_step):
        data_batch = self.preprocess_data(data_batch)
        x,_ = data_batch

        xhat = self.oops.pipeline.gibbs_sample(v=x,num_gibbs_steps=self.config.trainer.constrastive_diverge_sample_size)
        d = self.oops.model.logp_v_unnorm(x)
        m = self.oops.model.logp_v_unnorm(xhat)

        obj = d - m
        loss = -obj.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.writer.add_scalar('training loss', loss, number_of_training_step)
        return loss

    def test_step(self,data_batch):
        with torch.no_grad():
            data_batch = self.preprocess_data(data_batch)
            x, _ = data_batch

            xhat = self.oops.pipeline.gibbs_sample(v=x,num_gibbs_steps=self.config.trainer.constrastive_diverge_sample_size)
            d = self.oops.model.logp_v_unnorm(x)
            m = self.oops.model.logp_v_unnorm(xhat)

            obj = d - m
            loss_ = -obj.mean()
            return loss_

    def initialize(self):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(self.config.experiment_files.tensorboard_path)
        self.optimizer = Adam(self.oops.model.parameters(), lr=self.learning_rate)

        #========================
        # Initial Loss
        #========================
        data_batch = next(self.dataloader.train().__iter__())
        data_batch = self.preprocess_data(data_batch)
        x,_ = data_batch

        xhat = self.oops.pipeline.gibbs_sample(v=x,num_gibbs_steps=self.config.trainer.constrastive_diverge_sample_size)
        d = self.oops.model.logp_v_unnorm(x)
        m = self.oops.model.logp_v_unnorm(xhat)

        obj = d - m
        initial_loss = -obj.mean()

        assert torch.isnan(initial_loss).any() == False
        assert torch.isinf(initial_loss).any() == False

        self.save_results(ContrastiveDivergenceTrainerState(self.oops,
                                                            initial_loss=initial_loss,
                                                            average_test_loss=0.,
                                                            average_train_loss=0.,LOSS=[]),
                          epoch=0,checkpoint=True)

        return initial_loss

    def train(self):
        initial_loss = self.initialize()
        best_loss = initial_loss

        number_of_training_step = 0
        number_of_test_step = 0
        for epoch in range(self.number_of_epochs):
            state = ContrastiveDivergenceTrainerState(self.oops,
                                                      initial_loss,
                                                      average_train_loss=0,
                                                      average_test_loss=0,
                                                      LOSS=[])
            train_loss = []
            for data_batch in self.dataloader.train():
                loss = self.train_step(data_batch,number_of_training_step)
                train_loss.append(loss.item())
                state.LOSS.append(loss.item())
                number_of_training_step += 1
                if number_of_training_step % 10 == 0:
                    print("number_of_training_step: {}, Loss: {}".format(number_of_training_step, loss.item()))

            state.average_train_loss = np.asarray(train_loss).mean()

            test_loss = []
            for data_batch in self.dataloader.test():
                loss = self.test_step(data_batch)
                test_loss.append(loss.item())
                number_of_test_step+=1
            state.average_test_loss = np.asarray(test_loss).mean()

            # SAVE RESULTS IF LOSS DECREASES IN VALIDATION
            if state.average_test_loss  < best_loss:
                self.save_results(state,
                                  epoch,
                                  checkpoint = False)

            if (epoch + 1) % self.config.trainer.save_model_epochs == 0:
                self.save_results(state,
                                  epoch + 1,
                                  checkpoint=True)

        self.writer.close()

    def save_results(self,
                     state:ContrastiveDivergenceTrainerState,
                     epoch:int=0,
                     checkpoint:bool=False):
        RESULTS = {
            "model": state.oops.model,
            "initial_loss": state.initial_loss,
            "average_train_loss": state.average_train_loss,
            "average_test_loss": state.average_test_loss,
            "LOSS": state.LOSS
        }
        if checkpoint:
            torch.save(RESULTS,self.config.experiment_files.best_model_path_checkpoint.format(epoch))
        else:
            torch.save(RESULTS,self.config.experiment_files.best_model_path)