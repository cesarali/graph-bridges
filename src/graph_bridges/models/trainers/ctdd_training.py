





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