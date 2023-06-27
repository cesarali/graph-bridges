import os
import json

import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from dataclasses import asdict

from graph_bridges.models.backward_rates.backward_rate import GaussianTargetRateImageX0PredEMA
from graph_bridges.data.dataloaders import DoucetTargetData, GraphSpinsDataLoader
from graph_bridges.models.samplers.sampling import ReferenceProcess
from graph_bridges.models.reference_process.ctdd_reference import ReferenceProcess
from graph_bridges.models.losses.ctdd_losses import GenericAux
from graph_bridges.models.schedulers.scheduling_ctdd import CTDDScheduler
from graph_bridges.data.dataloaders_config import GraphSpinsDataLoaderConfig

from graph_bridges.models.pipelines.pipelines_utils import create_pipelines
from graph_bridges.models.schedulers.scheduling_utils import create_scheduler
from graph_bridges.models.backward_rates.backward_rate_utils import create_model
from graph_bridges.models.reference_process.reference_process_utils import create_reference
from graph_bridges.data.dataloaders_utils import create_dataloader
from graph_bridges.models.losses.loss_utils import create_loss

from graph_bridges.configs.graphs.lobster.config_mlp import BridgeMLPConfig
from graph_bridges.configs.graphs.lobster.config_base import BridgeConfig

if __name__=="__main__":
    #mlp_config = BridgeMLPConfig()
    config = BridgeConfig(experiment_indentifier=None)

    print(config.experiment_indentifier)
    print(config.results_dir)

    config.data = GraphSpinsDataLoaderConfig()
    #config.model = mlp_config.model
    device = torch.device(config.device)

    # =================================================================
    # CREATE OBJECTS FROM CONFIGURATION

    data_dataloader: GraphSpinsDataLoader
    model: GaussianTargetRateImageX0PredEMA
    reference_process: ReferenceProcess
    loss: GenericAux
    scheduler: CTDDScheduler

    data_dataloader = create_dataloader(config, device)
    model = create_model(config, device)
    reference_process = create_reference(config, device)
    loss = create_loss(config, device)
    scheduler = create_scheduler(config, device)

    # =================================================================
    minibatch = next(data_dataloader.train().__iter__())[0]

    optimizer = torch.optim.Adam(model.parameters(),
                                 config.optimizer.learning_rate)
    config.initialize()
    json.dump(asdict(config),
              open(config.config_path,"w"))

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(config.tensorboard_path)
    global_step = 0

    # Now you train the model
    for epoch in range(config.optimizer.num_epochs):
        progress_bar = tqdm(total=len(data_dataloader.train()))
        progress_bar.set_description(f"Epoch {epoch}")

        for step, minibatch in enumerate(data_dataloader.train()):
            minibatch = minibatch[0]
            B = minibatch.shape[0]

            # Sample a random timestep for each image
            ts = torch.rand((B,), device=device) * (1.0 - config.loss.min_time) + config.loss.min_time

            x_t, x_tilde, qt0, rate = scheduler.add_noise(minibatch,reference_process,ts,device,return_dict=False)
            x_logits, p0t_reg, p0t_sig, reg_x = model.forward(minibatch,ts,x_tilde)

            optimizer.zero_grad()
            loss_ = loss.calc_loss(minibatch,x_tilde,qt0,rate,x_logits,reg_x,p0t_sig,p0t_reg,device)
            loss_.backward()

            if config.optimizer.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            if config.model.do_ema:
                model.update_ema()

            # UPDATE LOGS
            progress_bar.update(1)
            logs = {"loss": loss_.detach().item(),
                    "global_step": global_step,
                    "step": global_step}
            writer.add_scalar("loss", loss_.detach().item(), global_step)
            progress_bar.set_postfix(**logs)

            # SAVE MODELS
            if global_step % config.optimizer.save_model_global_iter == 0:
                torch.save({"model":model.state_dict()},
                           config.best_model_path.format(0,global_step))

            # UPDATE STEPS
            global_step += 1
            if global_step > config.optimizer.max_n_iters:
                break

    writer.close()


