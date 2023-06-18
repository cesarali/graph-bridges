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

    from graph_bridges.models.samplers.sampling import ReferenceProcess
    from graph_bridges.data.dataloaders_utils import create_dataloader
    from graph_bridges.data.dataloaders import DoucetTargetData

    from graph_bridges.models.losses.ctdd_losses import GenericAux
    from graph_bridges.models.losses.loss_utils import create_loss

    from graph_bridges.configs.graphs.lobster.config_base import BridgeConfig
    from graph_bridges.models.backward_rates.backward_rate_utils import create_model
    from graph_bridges.models.reference_process.reference_process_utils import create_reference

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
    loss_ = loss.calc_loss(sample_,model,10)

    from graph_bridges.data.dataloaders_utils import create_dataloader


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