import torch
from torch.optim import Adam

import numpy as np
from pprint import pprint
from dataclasses import asdict

from ssda.data.dataloaders import NISTLoader

from ssda.models.vae_model import VAE
from ssda.configs.vae_config import VAEConfig
from ssda.losses.contrastive_loss import vae_loss
from ssda.models.encoder_config import EncoderConfig

class VAETrainer:

    name_="mutual_information_estimator"

    def __init__(self,
                 config: VAEConfig,
                 dataloader:NISTLoader=None,
                 vae:VAE=None):

        #set parameter values
        self.config = config
        self.learning_rate = config.trainer.learning_rate
        self.number_of_epochs = config.trainer.number_of_epochs
        self.device = torch.device(config.trainer.device)
        self.loss_type = config.trainer.loss_type

        #set other stuff
        if self.loss_type == "vae_loss":
            self.loss = vae_loss
        else:
            raise Exception("Loss Not Implemented")

        #define models
        self.vae = VAE()
        self.vae.create_new_from_config(self.config,self.device)
        self.dataloader = self.vae.dataloader


    def parameters_info(self):
        print("# ==================================================")
        print("# START OF BACKWARD MI TRAINING ")
        print("# ==================================================")
        print("# VAE parameters ************************************")
        pprint(asdict(self.vae.config))
        print("# Paths Parameters **********************************")
        pprint(asdict(self.dataloader.config))
        print("# Trainer Parameters")
        pprint(asdict(self.config))
        print("# ==================================================")
        print("# Number of Epochs {0}".format(self.number_of_epochs))
        print("# ==================================================")

    def preprocess_data(self,data_batch):
        return (data_batch[0].to(self.device), data_batch[1].to(self.device))

    def train_step(self,data_batch,number_of_training_step):

        data_batch = self.preprocess_data(data_batch)
        x,_ = data_batch
        recon_x, mu, logvar = self.vae(x)
        loss = self.loss(recon_x, x, mu, logvar)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.writer.add_scalar('training loss', loss, number_of_training_step)
        return loss

    def test_step(self,data_batch):
        with torch.no_grad():
            data_batch = self.preprocess_data(data_batch)
            x, _ = data_batch
            recon_x, mu, logvar = self.vae(x)
            loss_ = self.loss(recon_x, x, mu, logvar)
            return loss_

    def initialize(self):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(self.config.experiment_files.tensorboard_path)

        self.optimizer = Adam(self.vae.parameters(),lr=self.learning_rate)
        data_batch = next(self.dataloader.train().__iter__())
        data_batch = self.preprocess_data(data_batch)
        x,_ = data_batch
        recon_x, mu, logvar = self.vae(x)
        initial_loss = self.loss(recon_x, x, mu, logvar)

        assert torch.isnan(initial_loss).any() == False
        assert torch.isinf(initial_loss).any() == False

        self.save_results(self.vae,
                          initial_loss,
                          None,
                          None,
                          None,
                          0,
                          checkpoint=True)

        return initial_loss

    def train(self):
        initial_loss = self.initialize()
        best_loss = initial_loss

        number_of_training_step = 0
        number_of_test_step = 0
        for epoch in range(self.number_of_epochs):

            LOSS = []
            train_loss = []
            for data_batch in self.dataloader.train():
                loss = self.train_step(data_batch,number_of_training_step)
                train_loss.append(loss.item())
                LOSS.append(loss.item())
                number_of_training_step += 1
                if number_of_training_step % 100 == 0:
                    print("number_of_training_step: {}, Loss: {}".format(number_of_training_step, loss.item()))

            average_train_loss = np.asarray(train_loss).mean()

            test_loss = []
            for data_batch in self.dataloader.test():
                loss = self.test_step(data_batch)
                test_loss.append(loss.item())
                number_of_test_step+=1
            average_test_loss = np.asarray(test_loss).mean()

            # SAVE RESULTS IF LOSS DECREASES IN VALIDATION
            if average_test_loss < best_loss:
                self.save_results(self.vae,
                                  initial_loss,
                                  average_train_loss,
                                  average_test_loss,
                                  LOSS,
                                  epoch,
                                  checkpoint=False)

            if (epoch + 1) % self.config.trainer.save_model_epochs == 0:
                self.save_results(self.vae,
                                  initial_loss,
                                  average_train_loss,
                                  average_test_loss,
                                  LOSS,
                                  epoch+1,
                                  checkpoint=True)

        self.writer.close()

    def save_results(self,
                     binary_classifier,
                     initial_loss,
                     average_train_loss,
                     average_test_loss,
                     LOSS,
                     epoch=0,
                     checkpoint=False):
        if checkpoint:
            RESULTS = {
                "ssda":binary_classifier,
                "initial_loss":initial_loss,
                "average_train_loss":average_train_loss,
                "average_test_loss":average_test_loss,
                "LOSS":LOSS
            }
            torch.save(RESULTS,self.config.experiment_files.best_model_path_checkpoint.format(epoch))
        else:
            RESULTS = {
                "ssda":binary_classifier,
                "initial_loss":initial_loss,
                "average_train_loss":average_train_loss,
                "average_test_loss":average_test_loss,
                "LOSS":LOSS
            }
            torch.save(RESULTS,self.config.experiment_files.best_model_path)



