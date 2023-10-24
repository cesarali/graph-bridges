import os
import sys
import torch
from torch import nn
from torch import functional as F

import numpy as np
import pandas as pd

import sympy
from dataclasses import dataclass
from torchvision import transforms
from torch.optim.adam import Adam
from torch.distributions import Bernoulli
from torch.distributions import Categorical
from torch.nn.functional import softplus,softmax

from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Dirichlet
import torch

@dataclass
class Config:
    # data
    number_of_spins :int = 3
    number_of_states :int = 4
    sample_size :int = 200

    dirichlet_alpha_0 :float = 0.1
    dirichlet_alpha_1 :float = 100.

    bernoulli_probability_0 :float = 0.2
    bernoulli_probability_0 :float = 0.8

    # process
    gamma :float = .2

    # model

    # temporal network
    time_embed_dim :int = 9
    hidden_dim :int = 50

    # rate

    # training
    number_of_epochs = 100
    learning_rate = 0.01
    batch_size :int = 5
    device = "cuda:0"


from graph_bridges.models.temporal_networks.embedding_utils import transformer_timestep_embedding

class ConditionalBackwardRate(nn.Module):
    """

    """
    def __init__(self, config, device):
        super().__init__()
        self.expected_data_shape = [config.number_of_spins]
        self.temporal_network = TemporalMLP(dimensions=config.number_of_spins,
                                            number_of_states=config.number_of_states,
                                            time_embed_dim=config.time_embed_dim,
                                            hidden_dim=config.hidden_dim,
                                            device=device).to(device)
        # self.logits_to_rates = nn.Linear(self.temporal_network_output_size,)

    def forward(self, x, time):
        batch_size = x.size(0)

        # ================================
        expected_data_shape_ = torch.Size([batch_size] + self.expected_data_shape)
        temporal_network_logits = self.temporal_network(x, time)
        rates_ = softplus(temporal_network_logits)
        return rates_

class TemporalMLP(nn.Module):

    def __init__(self, dimensions, number_of_states, time_embed_dim, hidden_dim, device):
        super().__init__()

        self.time_embed_dim = time_embed_dim
        self.hidden_layer = hidden_dim
        self.num_states = number_of_states
        self.dimension = dimensions
        self.expected_output_shape = [self.dimension, self.num_states]

        self.define_deep_models()
        self.init_weights()

        self.device = device
        self.to(self.device)

    def define_deep_models(self):
        # layers
        self.f1 = nn.Linear(self.dimension, self.hidden_layer)
        self.f2 = nn.Linear(self.hidden_layer + self.time_embed_dim, self.dimension * self.num_states)

    def forward(self, x, times):
        batch_size = x.shape[0]
        time_embbedings = transformer_timestep_embedding(times,
                                                         embedding_dim=self.time_embed_dim)

        step_one = self.f1(x)
        step_two = torch.concat([step_one, time_embbedings], dim=1)
        rate_logits = self.f2(step_two)
        rate_logits = rate_logits.reshape(batch_size, self.dimension, self.num_states)

        return rate_logits

    def init_weights(self):
        nn.init.xavier_uniform_(self.f1.weight)
        nn.init.xavier_uniform_(self.f2.weight)

def sample_categorical_from_dirichlet(probs, alpha=None, sample_size=100, dimension=3, number_of_states=2):
    # ensure we have the probabilites
    if probs is None:
        if isinstance(alpha, float):
            alpha = torch.full((number_of_states,), alpha)
        else:
            assert len(alpha.shape) == 1
            assert alpha.size(0) == number_of_states
        # Sample from the Dirichlet distribution
        probs = Dirichlet(alpha).sample([sample_size])
    else:
        assert probs.max() <= 10.
        assert probs.max() >= 0.

    # Sample from the categorical distribution using the Dirichlet samples as probabilities
    categorical_samples = torch.multinomial(probs, dimension, replacement=True)
    return categorical_samples

def beta_integral(gamma, t1, t0):
    """
    Dummy integral for constant rate
    """
    interval = t1 - t0
    integral = gamma * interval
    return integral

def conditional_probability(config, x, x0, t, t0):
    """

    \begin{equation}
    P(x(t) = i|x(t_0)) = \frac{1}{s} + w_{t,t_0}\left(-\frac{1}{s} + \delta_{i,x(t_0)}\right)
    \end{equation}

    \begin{equation}
    w_{t,t_0} = e^{-S \int_{t_0}^{t} \beta(r)dr}
    \end{equation}

    """
    right_shape = lambda x: x if len(x.shape) == 3 else x[:, :, None]
    right_time_size = lambda t: t if isinstance(t, torch.Tensor) else torch.full((x.size(0),), t)

    t = right_time_size(t).to(x0.device)
    t0 = right_time_size(t0).to(x0.device)

    S = config.number_of_states
    integral_t0 = beta_integral(config.gamma, t, t0)

    w_t0 = torch.exp(-S * integral_t0)

    x = right_shape(x)
    x0 = right_shape(x0)

    delta_x = (x == x0).float()
    probability = 1. / S + w_t0[:, None, None] * ((-1. / S) + delta_x)

    return probability


def conditional_transition_probability(config, x, x1, x0, t):
    """
    \begin{equation}
    P(x_t=x|x_0,x_1) = \frac{p(x_1|x_t=x) p(x_t = x|x_0)}{p(x_1|x_0)}
    \end{equation}
    """

    P_x_to_x1 = conditional_probability(config, x1, x, t=1., t0=t)
    P_x0_to_x = conditional_probability(config, x, x0, t=t, t0=0.)
    P_x0_to_x1 = conditional_probability(config, x1, x0, t=1., t0=0.)

    conditional_transition_probability = (P_x_to_x1 * P_x0_to_x) / P_x0_to_x1
    return conditional_transition_probability


def constant_rate(config, x, t):
    right_time_size = lambda t: t if isinstance(t, torch.Tensor) else torch.full((x.size(0),), t)
    t = right_time_size(t).to(x.device)

    batch_size = x.size(0)
    dimension = x.size(1)

    assert batch_size == t.size(0)

    rate_ = torch.full((batch_size, dimension, config.number_of_states),
                       config.gamma)
    return rate_


def conditional_transition_rate(config, x, x1, t):
    """
    \begin{equation}
    f_t(\*x'|\*x,\*x_1) = \frac{p(\*x_1|x_t=\*x')}{p(\*x_1|x_t=\*x)}f_t(\*x'|\*x)
    \end{equation}
    """
    where_to_x = torch.arange(0, config.number_of_states)
    where_to_x = where_to_x[None, None, :].repeat((x.size(0), config.number_of_spins, 1)).float()
    where_to_x = where_to_x.to(x.device)

    P_xp_to_x1 = conditional_probability(config, x1, where_to_x, t=1., t0=t)
    P_x_to_x1 = conditional_probability(config, x1, x, t=1., t0=t)

    forward_rate = constant_rate(config, x, t).to(x.device)
    rate_transition = (P_xp_to_x1 / P_x_to_x1) * forward_rate

    return rate_transition


def uniform_pair_x0_x1(batch_1, batch_0):
    """
    Most simple Z sampler

    :param batch_1:
    :param batch_0:
    :return:
    """
    x_0 = batch_0[0]
    x_1 = batch_1[0]

    batch_size_0 = x_0.size(0)
    batch_size_1 = x_1.size(0)

    batch_size = min(batch_size_0, batch_size_1)

    x_0 = x_0[:batch_size, :]
    x_1 = x_1[:batch_size, :]

    return x_1, x_0

def sample_x(config,x_1, x_0, time):
    x_to_go = torch.arange(0, config.number_of_states)
    x_to_go = x_to_go[None, None, :].repeat((x_1.size(0), config.number_of_spins, 1)).float()
    x_to_go = x_to_go.to(device)

    prob_logits = conditional_transition_probability(config, x_to_go, x_1, x_0, time)
    probs = torch.softmax(prob_logits, dim=-1)
    sampled_x = Categorical(probs).sample().to(device).float()
    return sampled_x


config = Config()
if __name__=="__main__":

    # Parameters
    dataset_0 = sample_categorical_from_dirichlet(probs=None,
                                                  alpha=config.dirichlet_alpha_0,
                                                  sample_size=config.sample_size,
                                                  dimension=config.number_of_spins,
                                                  number_of_states=config.number_of_states)
    tensordataset_0 = TensorDataset(dataset_0)
    dataloader_0 = DataLoader(tensordataset_0, batch_size=config.batch_size)

    dataset_1 = sample_categorical_from_dirichlet(probs=None,
                                                  alpha=config.dirichlet_alpha_1,
                                                  sample_size=183,
                                                  dimension=config.number_of_spins,
                                                  number_of_states=config.number_of_states)
    tensordataset_1 = TensorDataset(dataset_1)
    dataloader_1 = DataLoader(tensordataset_1, batch_size=config.batch_size)




    device = torch.device(config.device) if torch.cuda.is_available() else torch.device("cpu")
    model = ConditionalBackwardRate(config, device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    number_of_training_steps = 0
    for epoch in range(config.number_of_epochs):
        for batch_1, batch_0 in zip(dataloader_1, dataloader_0):

            # data pair and time sample
            x_1, x_0 = uniform_pair_x0_x1(batch_1, batch_0)
            x_0 = x_0.float().to(device)
            x_1 = x_1.float().to(device)

            batch_size = x_0.size(0)
            time = torch.randn(batch_size).to(device)

            # sample x from z
            sampled_x = sample_x(config,x_1,x_0,time)

            # conditional rate
            conditional_rate = conditional_transition_rate(config, sampled_x, x_1, time)
            model_rate = model(sampled_x, time)

            optimizer.zero_grad()
            loss = (conditional_rate - model_rate).sum(axis=-1).sum(axis=-1)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            number_of_training_steps += 1

            if number_of_training_steps % 100 == 0:
                print(f"loss {round(loss.item(), 2)}")