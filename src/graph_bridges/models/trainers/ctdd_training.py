import torch
import numpy as np
from graph_bridges.models.trainers.training_utils import register_train_step
@register_train_step
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




if __name__=="__main__":
    from graph_bridges.models.backward_rates.backward_rate import GaussianTargetRateImageX0PredEMA
    from graph_bridges.data.dataloaders import DoucetTargetData
    from graph_bridges.models.samplers.sampling import ReferenceProcess
    from graph_bridges.models.reference_process.ctdd_reference import ReferenceProcess
    from graph_bridges.models.losses.ctdd_losses import GenericAux
    from graph_bridges.models.schedulers.scheduling_ctdd import CTDDScheduler

    from graph_bridges.models.pipelines.pipelines_utils import create_pipelines
    from graph_bridges.models.schedulers.scheduling_utils import create_scheduler
    from graph_bridges.models.backward_rates.backward_rate_utils import create_model
    from graph_bridges.models.reference_process.reference_process_utils import create_reference
    from graph_bridges.data.dataloaders_utils import create_dataloader
    from graph_bridges.models.losses.loss_utils import create_loss

    from graph_bridges.configs.graphs.lobster.config_base import BridgeConfig

    config = BridgeConfig()
    device = torch.device(config.device)

    #==================================================================
    # CREATE OBJECTS FROM CONFIGURATION

    data_dataloader: DoucetTargetData
    model : GaussianTargetRateImageX0PredEMA
    reference_process: ReferenceProcess
    loss : GenericAux
    scheduler:CTDDScheduler

    data_dataloader = create_dataloader(config,device)
    model = create_model(config,device)
    reference_process = create_reference(config,device)
    loss = create_loss(config,device)
    scheduler = create_scheduler(config,device)

    #=================================================================
    sample_ = data_dataloader.sample(config.number_of_paths, device)
    minibatch = sample_.unsqueeze(1).unsqueeze(1)

    # TIME ===========================================================
    #ts = torch.rand((minibatch.shape[0],), device=device) * (1.0 - config.loss.min_time) + config.loss.min_time
    B = minibatch.shape[0]
    ts = torch.rand((B,), device=device) * (1.0 - config.loss.min_time) + config.loss.min_time
    #==========

    x_t, x_tilde, qt0, rate = scheduler.add_noise(minibatch,reference_process,ts,device,return_dict=False)
    x_logits,p0t_reg,p0t_sig,reg_x = model.forward(minibatch,ts,x_tilde)
    loss_ = loss.calc_loss(minibatch,x_tilde,qt0,rate,x_logits,reg_x,p0t_sig,p0t_reg,device)

    print(loss_)

    """
    TRAINING EXAMPLE FOR DIFFUSERS LIBRARY
    https://huggingface.co/docs/diffusers/tutorials/basic_training
    
    clean_images = batch["images"]
    # Sample noise to add to the images
    noise = torch.randn(clean_images.shape).to(clean_images.device)
    bs = clean_images.shape[0]

    # Sample a random timestep for each image
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
    ).long()
    
    # CTDD original code ----------------------------------------------------------
    ts = torch.rand((B,), device=device) * (1.0 - self.min_time) + self.min_time
    # CTDD original code ----------------------------------------------------------

    # Add noise to the clean images according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
    
    with accelerator.accumulate(model):
        # Predict the noise residual
        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
        loss = F.mse_loss(noise_pred, noise)
        accelerator.backward(loss)
                
    """

    """
    while True:
        for minibatch in tqdm(dataloader):
            if cfg.data.name == "graphs":
                minibatch = minibatch[0]
            training_step.step(state, minibatch, loss, writer)

            if state['n_iter'] % cfg.saving.checkpoint_freq == 0 or state['n_iter'] == cfg.training.n_iters-1:
                bookkeeping.save_checkpoint(checkpoint_dir,
                                            state,
                                            cfg.saving.num_checkpoints_to_keep,
                                            cfg)

            if state['n_iter'] % cfg.saving.log_low_freq == 0 or state['n_iter'] == cfg.training.n_iters-1:
                for logger in low_freq_loggers:
                    logger(state=state, cfg=cfg, writer=writer,
                           minibatch=minibatch, dataset=dataset)

            state['n_iter'] += 1
            if state['n_iter'] > cfg.training.n_iters - 1:
                exit_flag = True
                break

        if exit_flag:
            break

    writer.close()
    return save_dir
    """

"""
    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
"""

"""
from diffusers.optimization import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)
"""

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