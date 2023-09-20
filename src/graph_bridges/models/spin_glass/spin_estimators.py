import os
import tqdm

import torch
from ising_parameters import ParametrizedIsingHamiltonian
from graph_bridges.utils.files_utils import create_dir_and_writer
from graph_bridges.models.spin_glass.spin_states_statistics import obtain_all_spin_states, obtain_new_spin_states
from torch.optim import Adam

g = lambda x: 1. / (1. + x)

def score_ratio(model, x_copy, x_flip):
    Q_copy = model(x_copy)
    Q_flipped = model(x_flip)
    loss = (g(Q_copy / Q_flipped)) ** 2.
    return loss

class ParametrizedIsingHamiltonianEstimator(ParametrizedIsingHamiltonian):

    def __init__(self):
        super(ParametrizedIsingHamiltonianEstimator).__init__()

    def oppers_estimator(self, states: torch.Tensor) -> torch.Tensor:
        """
        Here we evaluate over the difference between the states and the corresponding
        one spin flop configuration

        Parameters
        ----------
        IsingHamiltonian:ParametrizedIsingHamiltonian

        states:torch.Tensor

        Returns
        -------
        loss
        """
        new_states = obtain_new_spin_states(states,self.flip_mask)

        H_states = self(states)
        H_new_states = self(new_states)
        H_new_states = H_new_states.reshape(H_states.shape[0], self.number_of_spins)
        H = self.beta * (H_new_states - H_states[:, None])
        H = torch.exp(-.5 * H)
        loss = H.sum()
        return loss

    def inference(self,mcmc_sample:torch.Tensor,real_fields:torch.Tensor,real_couplings:torch.Tensor,
                  number_of_epochs=10000,learning_rate=1e-3):
        writer, results_path, best_model_path = create_dir_and_writer(model_name="oppers_estimation",
                                                                      experiments_class="",
                                                                      model_identifier="ising_{0}".format(self.model_identifier),
                                                                      delete=True)
        states = mcmc_sample[:,-1,:]
        optimizer = Adam(self.parameters(), lr=learning_rate)
        norm_loss_history = []
        oppers_loss_history = []
        for i in tqdm.tqdm(range(number_of_epochs)):
            loss = self.oppers_estimator(states)
            if real_couplings is not None:
                couplings_norm = torch.norm(real_couplings - self.couplings)
                norm_loss_history.append(couplings_norm.item())
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            oppers_loss_history.append(loss.item())

            writer.add_scalar("train/loss", loss.item(), i)
            writer.add_scalar("train/norm", couplings_norm.item(), i)
            print("train loss {0}".format(loss.item()))

        print("Saving Model In {0}".format(best_model_path))
        torch.save(self, best_model_path)
        torch.save({"real_fields": real_fields,
                    "real_couplings": real_couplings,
                    "paths": mcmc_sample,
                    "oppers_loss_history":oppers_loss_history,
                    "norm_loss_history":norm_loss_history},
                   os.path.join(results_path, "real_couplings.tr"))
        print("Real Data in {0}".format(results_path))
        return best_model_path, results_path

def total_loss_from_a_dataloader(model, spin_dataloader):
    number_of_spins = spin_dataloader.number_of_spins
    total_loss = 0.
    for x_sample in spin_dataloader:
        x_copy = x_sample.repeat_interleave(number_of_spins, dim=0)
        x_flip = obtain_new_spin_states(x_sample, model.flip_mask)

        loss = score_ratio(model, x_copy, x_flip).sum()
        total_loss += loss
    return total_loss