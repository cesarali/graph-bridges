import torch

def main():
    return None


if __name__=="__main__":
    from graph_bridges.models.backward_rates.backward_rate import GaussianTargetRateImageX0PredEMA

    from graph_bridges.data.dataloaders_utils import create_dataloader
    from graph_bridges.models.samplers.sampling import ReferenceProcess
    from graph_bridges.data.dataloaders import DoucetTargetData

    from graph_bridges.models.losses.ctdd_losses import GenericAux
    from graph_bridges.configs.graphs.lobster.config_base import BridgeConfig
    from graph_bridges.models.backward_rates.backward_rate_utils import create_model
    from graph_bridges.models.reference_process.reference_process_utils import create_reference
    from graph_bridges.models.losses.loss_utils import create_loss

    config = BridgeConfig()
    device = torch.device(config.device)

    data_dataloader: DoucetTargetData
    model : GaussianTargetRateImageX0PredEMA
    loss : GenericAux

    print(config.loss.name)

    data_dataloader = create_dataloader(config,device)
    model = create_model(config,device)
    reference_process = create_reference(config,device)
    loss = create_loss(config,device)

    sample_ = data_dataloader.sample(config.number_of_paths, device)

    time = torch.full((config.number_of_paths,),
                      config.sampler.min_t)
    forward = model(sample_,time)


    sample_ = sample_.unsqueeze(1).unsqueeze(1)
    loss.calc_loss(sample_,model,10)






# clean_images = batch["images"]
# Sample noise to add to the images
# # noise = torch.randn(clean_images.shape).to(clean_images.device)
# # bs = clean_images.shape[0]

# Sample a random timestep for each image
# timesteps = torch.randint(
    #0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
#).long()

# Add noise to the clean images according to the noise magnitude at each timestep
# (this is the forward diffusion process)
#noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)