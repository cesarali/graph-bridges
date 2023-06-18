import os
import tqdm
import time
import torch

from torch import nn
from pprint import pprint
from torch.optim import Adam
import numpy as np
from torch.distributions import Normal, Bernoulli

from discrete_diffusion.models.ising.spin_states_statistics import obtain_all_spin_states, obtain_new_spin_states
from discrete_diffusion.utils.basic_setups import create_dir_and_writer

class bernoulli_spins():
    """
    Spin probabilities based on bernoulli distribution
    per site
    """
    def __init__(self,p):
        self.p = p
        self.bernoulli_distribution = Bernoulli(p)

    def sample(self,sample_shape):
        sample_one_zeros = self.bernoulli_distribution.sample(sample_shape=sample_shape)
        sample_one_zeros[torch.where(sample_one_zeros == 0)] = -1.
        return sample_one_zeros

def initialize_model(number_of_spins,number_of_paths,J_mean,J_std):
    # define random interactions (fields in the diagonal)
    J = Normal(J_mean,J_std)
    J = J.sample(sample_shape=(number_of_spins,number_of_spins))
    J = (J + J.T)*.5

    # define uniform spin configurations
    x = torch.randint(0,2,(number_of_paths,number_of_spins))
    x[torch.where(x==0)] = -1
    x = x.float()
    paths = x.unsqueeze(1)

    return paths, J

class ParametrizedIsingHamiltonian(nn.Module):
    """
    Simple Hamiltonian Model for Parameter Estimation
    """
    def __init__(self, **kwargs):
        """
        Parameters
        ----------

        number_of_spins: int
        obtain_partition_function: bool
            only defined for number_of_spins < 10
        """
        super(ParametrizedIsingHamiltonian, self).__init__()
        self.beta = kwargs.get("beta")
        self.number_of_spins = kwargs.get("number_of_spins")

        self.lower_diagonal_indices = torch.tril_indices(self.number_of_spins, self.number_of_spins, -1)
        self.number_of_couplings = self.lower_diagonal_indices[0].shape[0]

        self.flip_mask = torch.ones((self.number_of_spins, self.number_of_spins))
        self.flip_mask.as_strided([self.number_of_spins], [self.number_of_spins + 1]).copy_(torch.ones(self.number_of_spins) * -1.)
        self.model_identifier = str(int(time.time()))
        self.define_parameters(**kwargs)
        self.model_parameters = kwargs

    def obtain_parameters(self):
        return self.model_parameters

    @classmethod
    def coupling_size(self,number_of_spins):
        lower_diagonal_indices = torch.tril_indices(number_of_spins, number_of_spins, -1)
        number_of_couplings = lower_diagonal_indices[0].shape[0]
        return number_of_couplings

    @classmethod
    def sample_random_model(self,number_of_spins,couplings_sigma=None):
        number_of_couplings = self.coupling_size(number_of_spins)

        if couplings_sigma is None:
            couplings_sigma = 1/float(number_of_spins)

        couplings = torch.clone(torch.Tensor(size=(
            number_of_couplings,)).normal_(0., couplings_sigma))
        fields = torch.clone(torch.Tensor(size=(
            number_of_spins,)).normal_(0., 1 / couplings_sigma))

        return fields,couplings

    def define_parameters(self,**kwargs):
        couplings = kwargs.get("couplings")
        fields = kwargs.get("fields")

        self.couplings_deterministic = kwargs.get("couplings_deterministic")
        self.couplings_sigma = kwargs.get("couplings_sigma")

        # INITIALIZING COUPLINGS AND FIELDS FROM ARRAYS
        if couplings is not None and fields is not None:
            assert isinstance(couplings,(list,torch.Tensor))
            assert isinstance(fields,(list,torch.Tensor))

            if isinstance(couplings,(list,)):
                couplings = torch.Tensor(couplings)
            if isinstance(fields,(list,)):
                fields = torch.Tensor(fields)

            assert self.number_of_couplings == couplings.shape[0]
            assert self.number_of_spins == fields.shape[0]
        else:
            # INITIALIZING COUPLINGS AND FIELDS FROM FLOAT
            if self.couplings_deterministic is None:
                fields,couplings = self.sample_random_model(number_of_spins, self.couplings_sigma)
            else:
                if isinstance(self.couplings_deterministic,(float)):
                    couplings = torch.full((self.number_of_couplings,),self.couplings_deterministic)
                    fields = torch.full((self.number_of_spins,),self.couplings_deterministic)

        #CONVERT INTO PARAMETERS
        self.fields = nn.Parameter(fields)
        self.couplings = nn.Parameter(couplings)
        #========================================================================
        if kwargs.get("obtain_partition_function"):
            self.obtain_partition_function()
        else:
            self.partition_function = None

    def obtain_partition_function(self):
        assert self.number_of_spins < 12
        all_states = obtain_all_spin_states(self.number_of_spins)
        with torch.no_grad():
            self.partition_function = self(all_states)
            self.partition_function = torch.exp(-self.partition_function).sum()

    def obtain_couplings_as_matrix(self,couplings=None):
        """
        converts the parameters vectors which store only the lower diagonal
        into a full symetric matrix

        :return:
        """
        if couplings is None:
            couplings = self.couplings
        coupling_matrix = torch.zeros((self.number_of_spins, self.number_of_spins))
        coupling_matrix[self.lower_diagonal_indices[0], self.lower_diagonal_indices[1]] = couplings
        coupling_matrix = coupling_matrix + coupling_matrix.T
        return coupling_matrix

    def log_probability(self,states,average=True):
        """
        :return:
        """
        if self.partition_function is not None:
            if average:
                return -self.beta*self(states).mean() + torch.log(self.partition_function)
            else:
                return -self.beta * self(states).std() + torch.log(self.partition_function)
        else:
            if average:
                return -self.beta * self(states).mean()
            else:
                return -self.beta * self(states).std()

    def sample(self,number_of_paths:int,number_of_mcmc_steps: int, **kwargs)-> torch.Tensor:
        """
        Here we follow a basic metropolis hasting algorithm

        :return:
        """
        with torch.no_grad():
            self.number_of_paths = number_of_paths
            self.number_of_mcmc_steps = number_of_mcmc_steps

            # we start a simple bernoulli spin distribution
            p0 = torch.Tensor(self.number_of_spins * [0.5])
            initial_distribution = bernoulli_spins(p0)
            states = initial_distribution.sample(sample_shape=(self.number_of_paths,))
            paths = states.unsqueeze(1)
            rows_index = torch.arange(0, self.number_of_paths)

            # METROPOLIS HASTING
            for mcmc_index in range(self.number_of_mcmc_steps):
                i_random = torch.randint(0, self.number_of_spins, (self.number_of_paths,))
                index_to_change = (rows_index, i_random)

                states = paths[:,-1,:]
                new_states = torch.clone(states)
                new_states[index_to_change] = states[index_to_change] * -1.

                H_0 = self(states)
                H_1 = self(new_states)
                H = self.beta * (H_0-H_1)

                flip_probability = torch.exp(H)
                r = torch.rand((number_of_paths,))
                where_to_flip = r < flip_probability

                new_states = torch.clone(states)
                index_to_change = (rows_index[torch.where(where_to_flip)], i_random[torch.where(where_to_flip)])
                new_states[index_to_change] = states[index_to_change] * -1.

                paths = torch.cat([paths, new_states.unsqueeze(1)], dim=1)

            return paths

    def hamiltonian_diagonal(self, states, i_random):
        coupling_matrix = self.obtain_couplings_as_matrix()
        J_i = coupling_matrix[:, i_random].T
        H_i = self.fields[i_random]
        H_i =  H_i + torch.einsum('bi,bi->b', J_i, states)
        return H_i

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Ising Hamiltonian

        Parameters
        ----------
        states:torch.Tensor
            Ising states defined by 1 or -1 spin values

        Returns
        -------
        Hamiltonian
        """
        coupling_matrix = self.obtain_couplings_as_matrix()
        H_couplings = torch.einsum('bi,bij,bj->b', states, coupling_matrix[None, :, :], states)
        H_fields = torch.einsum('bi,bi->b', self.fields[None, :], states)
        Hamiltonian = - H_couplings - H_fields
        return Hamiltonian

    @classmethod
    def get_parameters(self):
        kwargs = {
            "number_of_spins": 4,
            "beta": 1.,
            "obtain_partition_function": True,
            "couplings_deterministic":1.,
            "couplings_sigma":1.,
            "couplings":None,
            "fields":None,
            "number_of_paths": 2,
            "number_of_mcmc_steps": 1000,
        }
        return kwargs


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

g = lambda x: 1. / (1. + x)

def score_ratio(model, x_copy, x_flip):
    Q_copy = model(x_copy)
    Q_flipped = model(x_flip)
    loss = (g(Q_copy / Q_flipped)) ** 2.
    return loss

def total_loss_from_a_dataloader(model, spin_dataloader):
    total_loss = 0.
    for x_sample in spin_dataloader:
        x_copy = x_sample.repeat_interleave(number_of_spins, dim=0)
        x_flip = obtain_new_spin_states(x_sample, model.flip_mask)

        loss = score_ratio(model, x_copy, x_flip).sum()
        total_loss += loss
    return total_loss

#=============================================================
# ARGUMENTS
#=============================================================

if __name__=="__main__":
    from discrete_diffusion.models.ising.spin_states_statistics import log_likelihood_of_path
    from discrete_diffusion.data.datasets import BasicDataSet, DataLoader

    #===========================================================================
    # TEST PARAMETRIC MODEL
    #===========================================================================
    number_of_spins = 100
    number_of_paths = 3000
    batch_size = 32
    number_of_mcmc_steps = 5000

    ising_parameters = ParametrizedIsingHamiltonian.get_parameters()
    number_of_couplings = ParametrizedIsingHamiltonian.coupling_size(number_of_spins)

    #fields = torch.full((number_of_spins,),0.2)
    #couplings = torch.full((number_of_couplings,),.1)

    fields = torch.Tensor(size=(number_of_spins,)).normal_(0.,1./number_of_spins)
    couplings = torch.Tensor(size=(number_of_couplings,)).normal_(0.,1/number_of_spins)

    #fields, couplings = ParametrizedIsingHamiltonian.sample_random_model(number_of_spins)

    pprint(ising_parameters)
    ising_parameters.update({"number_of_spins":number_of_spins,
                             'couplings_deterministic': None,
                             'obtain_partition_function':False,
                             "number_of_mcmc_steps":number_of_mcmc_steps,
                             'couplings_sigma': 5.,
                             "fields": fields,
                             "couplings": couplings})

    ising_model_real = ParametrizedIsingHamiltonian(**ising_parameters)
    ising_mcmc_sample = ising_model_real.sample(**ising_parameters)
    ising_sample = ising_mcmc_sample[:, 500, :]


    log_likelihood_of_path(ising_model_real,ising_mcmc_sample,plot=True)
    real_couplings = torch.clone(ising_model_real.couplings)
    real_fields = torch.clone(ising_model_real.fields)

    # Create the dataset and the model
    #ising_dataset = BinaryDataSet(ising_sample)
    #ising_dataloader = DataLoader(ising_dataset,
    #                              batch_size=batch_size,
    #                              shuffle=True)
    #print(next(ising_dataloader.__iter__()).shape)

    #===========================================================================
    # TEST MANFRED ESTIMATOR
    #===========================================================================
    #kwargs.update({"couplings_deterministic": None, "couplings_sigma": None})
    #PIH_train = ParametrizedIsingHamiltonian(**kwargs)
    #PIH_train.inference(mcmc_sample,real_fields,real_couplings,
    #                    number_of_epochs= 10000,learning_rate = 1e-3)
    #===========================================================================
    # TEST DYNAMICS
    #===========================================================================
    """
    #paths, J = initialize_model(number_of_spins,number_of_paths,J_mean,J_std)
    #paths = glauber_dynamics(T, tau, paths, J)

    time_step = 0
    depth_index = 0
    #x = paths[:, time_step, :]
    #x = lexicographical_ordering_of_spins(x)

    initial_distribution = bernoulli_spins(p0)
    final_distribution = bernoulli_spins(p1)

    PHI_0 = MLP(**phi_parameters)
    PHI_1 = MLP(**phi_parameters)

    time_grid = torch.arange(0., T, tau)
    paths = initial_distribution.sample([number_of_paths]).unsqueeze(1)
    paths,times = parametric_dynamics(paths, time_grid, number_of_paths, number_of_spins, time_embedding_dim, PHI_1)
    """