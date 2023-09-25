import os
import torch
from dataclasses import dataclass
from torch.distributions import Normal
from torch.utils.data import TensorDataset,DataLoader

def generate_path(initial_x,timesteps):
    path = [initial_x.unsqueeze(1)]
    for time in timesteps[1:]:
        noise = noise_distribution.sample()
        path.append((initial_x + noise).unsqueeze(1))
    path = torch.concat(path,axis=1)
    return path

def create_paths_generator(dataloader,timesteps):
    for databatch in dataloader:
        initial_x = databatch[0]
        batch_size = initial_x.shape[0]
        paths = generate_path(initial_x,timesteps)
        timesteps_ = timesteps.repeat(batch_size,1)

        batch_size,number_of_time_steps,number_of_spins = paths.shape
        paths = paths.reshape(batch_size*number_of_time_steps,number_of_spins)
        timesteps_ = timesteps_.reshape(batch_size*number_of_time_steps)

        perm = torch.randperm(paths.shape[0])
        paths = paths[perm]
        timesteps_ = timesteps_[perm]
        chunks_ = torch.chunk(paths,number_of_time_steps)
        time_chunks = torch.chunk(timesteps_,number_of_time_steps)
        for chunk in zip(chunks_,time_chunks):
            yield chunk

if __name__=="__main__":
    batch_size = 4
    number_of_steps = 8
    number_of_spins = 10
    training_size = 103

    delta_t = 1. / number_of_steps
    timesteps = torch.linspace(0., 1., number_of_steps + 1)

    initial_distribution = Normal(torch.zeros(number_of_spins, ),
                                  torch.ones(number_of_spins, ))

    noise_distribution = Normal(torch.zeros(number_of_spins, ),
                                torch.ones(number_of_spins, ))

    total_data = initial_distribution.sample((training_size,))
    dataloader = DataLoader(TensorDataset(total_data), batch_size=batch_size)
    initial_x = next(dataloader.__iter__())[0]

    number_of_runs = 0
    for dynamic_batch in create_paths_generator(dataloader,timesteps):
        print(f"Batch Size {dynamic_batch[0].shape[0]}")
        number_of_runs += dynamic_batch[0].shape[0]
    print(number_of_runs)