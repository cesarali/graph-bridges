import torch
from torch.utils.data import TensorDataset,DataLoader

def sample_categorical_from_dirichlet(probs, alpha=None, sample_size=100, dimension=3, number_of_states=2):
    # ensure we have the probabilites
    if probs is None:
        if isinstance(alpha, float):
            alpha = torch.full((number_of_states,), alpha)
        else:
            assert len(alpha.shape) == 1
            assert alpha.size(0) == number_of_states
        # Sample from the Dirichlet distribution
        probs = torch.distributions.Dirichlet(alpha).sample([sample_size])
    else:
        assert probs.max() <= 1.
        assert probs.max() >= 0.

    # Sample from the categorical distribution using the Dirichlet samples as probabilities
    categorical_samples = torch.multinomial(probs, dimension, replacement=True)
    return categorical_samples

if __name__=="__main__":
    from dataclasses import dataclass

    @dataclass
    class Config:
        # data
        number_of_spins: int = 3
        number_of_states: int = 2
        sample_size: int = 200

        dirichlet_alpha_0 = 0.9
        dirichlet_alpha_1 = 100.

        bernoulli_probability_0 = 0.2
        bernoulli_probability_1 = 0.8

        # process
        gamma: float = .2

        # training
        number_of_epochs = 1
        learning_rate = 0.01
        batch_size: int = 3


    config = Config()

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
                                                  sample_size=300,
                                                  dimension=config.number_of_spins,
                                                  number_of_states=config.number_of_states)
    tensordataset_1 = TensorDataset(dataset_1)
    dataloader_1 = DataLoader(tensordataset_1, batch_size=config.batch_size)

    dataloader_sample_size_0 = 0.
    dataloader_sample_size_1 = 0.
    for epoch in range(config.number_of_epochs):
        for batch_1, batch_0 in zip(dataloader_1, dataloader_0):
            # Unpack your data from each batch if needed
            # For example, if each batch is a tuple of (data, label):
            dataloader_sample_size_0 += batch_0[0].size(0)
            dataloader_sample_size_1 += batch_1[0].size(0)

    print(dataloader_sample_size_0)
    print(dataloader_sample_size_1)