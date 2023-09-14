import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from abc import ABC,abstractmethod
from graph_bridges.models.networks import networks_tau
from graph_bridges.configs.graphs.graph_config_sb import SBConfig

from torchtyping import TensorType
from graph_bridges.models.networks.embedding_utils import transformer_timestep_embedding
from graph_bridges.models.reference_process.ctdd_reference import GaussianTargetRate
from typing import Union, Tuple

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from graph_bridges.models.networks.ema import EMA
from graph_bridges.models.backward_rates.backward_rate import BackwardRate
from graph_bridges.models.networks.network_utils import load_temp_network

class ImageX0PredBase(BackwardRate):
    def __init__(self, cfg, device, rank=None):
        BackwardRate.__init__(self,cfg,device,rank)
        self.fix_logistic = cfg.model.fix_logistic

        tmp_net = load_temp_network(config=cfg, device=device)

        if cfg.distributed:
            self.net = DDP(tmp_net, device_ids=[rank])
        else:
            self.net = tmp_net

        self.S = cfg.data.S
        self.data_shape = cfg.data.shape
        self.device = device

    def _forward(self,
        x: TensorType["B", "D"],
        times: TensorType["B"]
    ) -> TensorType["B", "D", "S"]:
        """
            Returns logits over state space for each pixel
        """
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            x = x.view(B, C*H*W)

        B, D = x.shape
        C,H,W = self.data_shape
        S = self.S
        x = x.view(B, C, H, W)

        net_out = self.net(x, times) # (B, 2*C, H, W)

        # Truncated logistic output from https://arxiv.org/pdf/2107.03006.pdf

        mu = net_out[:, 0:C, :, :].unsqueeze(-1)
        log_scale = net_out[:, C:, :, :].unsqueeze(-1)

        inv_scale = torch.exp(- (log_scale - 2))

        bin_width = 2. / self.S
        bin_centers = torch.linspace(start=-1. + bin_width/2,
            end=1. - bin_width/2,
            steps=self.S,
            device=self.device).view(1, 1, 1, 1, self.S)

        sig_in_left = (bin_centers - bin_width/2 - mu) * inv_scale
        bin_left_logcdf = F.logsigmoid(sig_in_left)
        sig_in_right = (bin_centers + bin_width/2 - mu) * inv_scale
        bin_right_logcdf = F.logsigmoid(sig_in_right)

        logits_1 = self._log_minus_exp(bin_right_logcdf, bin_left_logcdf)
        logits_2 = self._log_minus_exp(-sig_in_left + bin_left_logcdf, -sig_in_right + bin_right_logcdf)
        if self.fix_logistic:
            logits = torch.min(logits_1, logits_2)
        else:
            logits = logits_1

        logits = logits.view(B,D,S)

        return logits

    def _log_minus_exp(self, a, b, eps=1e-6):
        """
            Compute log (exp(a) - exp(b)) for (b<a)
            From https://arxiv.org/pdf/2107.03006.pdf
        """
        return a + torch.log1p(-torch.exp(b-a) + eps)

class BackRateMLP(EMA,BackwardRate,GaussianTargetRate):

    def __init__(self,config,device,rank=None):
        EMA.__init__(self,config)
        BackwardRate.__init__(self,config,device,rank)

        self.hidden_layer = config.model.hidden_layer
        self.define_deep_models()
        #self.init_parameters()
        self.init_ema()
        self.to(device)

    def define_deep_models(self):
        # layers
        self.f1 = nn.Linear(self.dimension, self.hidden_layer)
        self.f2 = nn.Linear(self.hidden_layer + self.time_embed_dim,self.dimension*self.num_states)

        # temporal encoding
        #self.temb_modules = []
        #self.temb_modules.append(nn.Linear(self.time_embed_dim, self.time_embed_dim * 4))
        #nn.init.zeros_(self.temb_modules[-1].bias)
        #self.temb_modules.append(nn.Linear(self.time_embed_dim * 4, self.time_embed_dim * 4))
        #nn.init.zeros_(self.temb_modules[-1].bias)
        #self.temb_modules = nn.ModuleList(self.temb_modules)

        #self.expanded_time_dim = 4 * self.time_embed_dim if self.do_time_embed else None

    def _forward(self,
                x: TensorType["batch_size", "dimension"],
                times: TensorType["batch_size"]
                ) -> TensorType["batch_size", "dimension", "num_states"]:

        if self.config.data.type == "doucet":
            x = self._center_data(x)

        batch_size = x.shape[0]
        time_embbedings = transformer_timestep_embedding(times,
                                                         embedding_dim=self.time_embed_dim)

        step_one = self.f1(x)
        step_two = torch.concat([step_one, time_embbedings], dim=1)
        rate_logits = self.f2(step_two)
        rate_logits = rate_logits.reshape(batch_size,self.dimension,self.num_states)
        return rate_logits

    def init_parameters(self):
        nn.init.xavier_uniform_(self.f1.weight)
        nn.init.xavier_uniform_(self.f2.weight)

class BackRateConstant(EMA,BackwardRate,GaussianTargetRate):

    def __init__(self,config,device,rank=None,constant=10.):
        EMA.__init__(self,config)
        BackwardRate.__init__(self,config,device,rank)
        self.constant = constant
        self.hidden_layer = config.model.hidden_layer
        self.define_deep_models()
        #self.init_parameters()
        self.init_ema()

    def define_deep_models(self):
        # layers
        self.f1 = nn.Linear(self.dimension, self.hidden_layer)
        self.f2 = nn.Linear(self.hidden_layer + self.time_embed_dim,self.dimension*self.num_states)

        # temporal encoding
        #self.temb_modules = []
        #self.temb_modules.append(nn.Linear(self.time_embed_dim, self.time_embed_dim * 4))
        #nn.init.zeros_(self.temb_modules[-1].bias)
        #self.temb_modules.append(nn.Linear(self.time_embed_dim * 4, self.time_embed_dim * 4))
        #nn.init.zeros_(self.temb_modules[-1].bias)
        #self.temb_modules = nn.ModuleList(self.temb_modules)

        #self.expanded_time_dim = 4 * self.time_embed_dim if self.do_time_embed else None

    def _forward(self,
                x: TensorType["batch_size", "dimension"],
                times: TensorType["batch_size"]
                ) -> TensorType["batch_size", "dimension", "num_states"]:

        if self.config.data.type == "doucet":
            x = self._center_data(x)

        batch_size = x.shape[0]

        return torch.full(torch.Size([batch_size,self.dimension,self.num_states]),self.constant)

    def init_parameters(self):
        pass

class GaussianTargetRateImageX0PredEMA(EMA,ImageX0PredBase):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        ImageX0PredBase.__init__(self, cfg, device, rank)
        #GaussianTargetRate.__init__(self,cfg, device)
        self.config = cfg
        self.init_ema()

from graph_bridges.models.networks.network_utils import load_temp_network

from graph_bridges.models.networks.transformers.temporal_hollow_transformers import TemporalHollowTransformer

class BackwardRateTemporalHollowTransformer(EMA,BackwardRate,GaussianTargetRate):

    def __init__(self,config,device,rank=None):
        EMA.__init__(self,config)
        BackwardRate.__init__(self,config,device,rank)

        self.define_deep_models(config,device)
        #self.init_parameters()
        self.init_ema()
        self.to(device)

    def define_deep_models(self,config,device):
        self.temporal_hollow_transformers: TemporalHollowTransformer
        self.temporal_hollow_transformers = load_temp_network(config=config, device=device)

    def _forward(self,
                x: TensorType["batch_size", "dimension"],
                times: TensorType["batch_size"]
                ) -> TensorType["batch_size", "dimension", "num_states"]:

        tht = self.temporal_hollow_transformers(x.long(),times)
        return tht

    def init_parameters(self):
        nn.init.xavier_uniform_(self.f1.weight)
        nn.init.xavier_uniform_(self.f2.weight)


all_backward_rates = {"BackRateConstant":BackRateConstant,
                      "BackRateMLP":BackRateMLP,
                      "GaussianTargetRateImageX0PredEMA":GaussianTargetRateImageX0PredEMA}