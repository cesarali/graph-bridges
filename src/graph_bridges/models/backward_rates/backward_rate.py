import math
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from dataclasses import dataclass,asdict,field

from abc import ABC,abstractmethod
from graph_bridges.models.backward_rates import backward_rate_utils
from graph_bridges.models.networks_arquitectures import networks
from graph_bridges.configs.graphs.lobster.config_base import BridgeConfig

from torchtyping import TensorType
from graph_bridges.models.networks_arquitectures.network_utils import transformer_timestep_embedding
from torch.nn.functional import softplus,softmax
from graph_bridges.models.reference_process.ctdd_reference import GaussianTargetRate
from typing import Tuple, Union
from typing import List, Union, Optional, Tuple


from dataclasses import dataclass
from diffusers.utils import BaseOutput

@dataclass
class BackwardRateOutput(BaseOutput):
    """
    :param BaseOutput:
    :return:
    """
    x_logits: torch.Tensor
    p0t_reg: torch.Tensor
    p0t_sig: torch.Tensor
    reg_x: torch.Tensor

class BackwardRate(nn.Module,ABC):

    def __init__(self,
                 config:BridgeConfig,
                 device,
                 rank,
                 **kwargs):
        super().__init__()

        self.config = config

        # DATA
        self.dimension = config.data.D
        self.num_states = config.data.S
        #self.data_type = config.data.type
        self.data_min_max = config.data.data_min_max

        # TIME
        self.time_embed_dim = config.model.time_embed_dim
        self.act = nn.functional.silu

    def init_parameters(self):
        return None

    @abstractmethod
    def _forward(self,
                x: TensorType["batch_size", "dimension"],
                times:TensorType["batch_size"]
                )-> TensorType["batch_size", "dimension", "num_states"]:
        return None

    def _center_data(self, x):
        out = (x - self.data_min_max[0]) / (self.data_min_max[1] - self.data_min_max[0])  # [0, 1]
        return 2 * out - 1  # to put it in [-1, 1]

    def forward(self,
                x: TensorType["batch_size", "dimension"],
                times:TensorType["batch_size"],
                x_tilde: TensorType["batch_size", "dimension"] = None,
                return_dict: bool = False,
                )-> Union[BackwardRateOutput, torch.FloatTensor, Tuple]:
        #if self.data_type == "doucet":
        if x_tilde is not None:
            return self.ctdd(x,x_tilde,times,return_dict)
        else:
            h = x
            x_logits = self._forward(h, times)
            if not return_dict:
                return x_logits
            else:
                return BackwardRateOutput(x_logits=x_logits, p0t_reg=None, p0t_sig=None)

    def ctdd(self,x_t,x_tilde,times,return_dict):
        if x_tilde is not None:
            if self.config.loss.one_forward_pass:
                reg_x = x_tilde
                x_logits = self._forward(reg_x, times)  # (B, D, S)
                p0t_reg = F.softmax(x_logits, dim=2)  # (B, D, S)
                p0t_sig = p0t_reg
            else:
                reg_x = x_t
                x_logits = self._forward(reg_x, times)  # (B, D, S)
                p0t_reg = F.softmax(x_logits, dim=2)  # (B, D, S)
                p0t_sig = F.softmax(self._forward(x_tilde, times), dim=2)  # (B, D, S)

            if not return_dict:
                return (x_logits,p0t_reg,p0t_sig,reg_x)
            return BackwardRateOutput(x_logits=x_logits,p0t_reg=p0t_reg,p0t_sig=p0t_sig,reg_x=reg_x)
        else:
            return self._forward(x_t, times)  # (B, D, S)


    def stein_binary_forward(self,
                x: TensorType["batch_size", "dimension"],
                times: TensorType["batch_size"]
                ) -> TensorType["batch_size", "dimension"]:
        forward_logits = self.forward(x,times)
        return forward_logits[:,:,0]

class ImageX0PredBase(BackwardRate):
    def __init__(self, cfg, device, rank=None):
        BackwardRate.__init__(self,cfg,device,rank)

        self.fix_logistic = cfg.model.fix_logistic
        ch = cfg.model.ch
        num_res_blocks = cfg.model.num_res_blocks
        num_scales = cfg.model.num_scales
        ch_mult = cfg.model.ch_mult
        input_channels = cfg.model.input_channels
        output_channels = cfg.model.input_channels * cfg.data.S
        scale_count_to_put_attn = cfg.model.scale_count_to_put_attn
        data_min_max = cfg.model.data_min_max
        dropout = cfg.model.dropout
        skip_rescale = cfg.model.skip_rescale
        do_time_embed = True
        time_scale_factor = cfg.model.time_scale_factor
        time_embed_dim = cfg.model.time_embed_dim

        tmp_net = networks.UNet(
                ch, num_res_blocks, num_scales, ch_mult, input_channels,
                output_channels, scale_count_to_put_attn, data_min_max,
                dropout, skip_rescale, do_time_embed, time_scale_factor,
                time_embed_dim
        ).to(device)
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

class EMA():
    def __init__(self, cfg):
        self.decay = cfg.model.ema_decay
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.shadow_params = []
        self.collected_params = []
        self.num_updates = 0

    def init_ema(self):
        self.shadow_params = [p.clone().detach()
                            for p in self.parameters() if p.requires_grad]

    def update_ema(self):

        if len(self.shadow_params) == 0:
            raise ValueError("Shadow params not initialized before first ema update!")

        decay = self.decay
        self.num_updates += 1
        decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in self.parameters() if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def state_dict(self):
        sd = nn.Module.state_dict(self)
        sd['ema_decay'] = self.decay
        sd['ema_num_updates'] = self.num_updates
        sd['ema_shadow_params'] = self.shadow_params

        return sd

    def move_shadow_params_to_model_params(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def move_model_params_to_collected_params(self):
        self.collected_params = [param.clone() for param in self.parameters()]

    def move_collected_params_to_model_params(self):
        for c_param, param in zip(self.collected_params, self.parameters()):
            param.data.copy_(c_param.data)

    def load_state_dict(self, state_dict):
        missing_keys, unexpected_keys = nn.Module.load_state_dict(self, state_dict, strict=False)

        # print("state dict keys")
        # for key in state_dict.keys():
        #     print(key)

        if len(missing_keys) > 0:
            print("Missing keys: ", missing_keys)
            raise ValueError
        if not (len(unexpected_keys) == 3 and \
            'ema_decay' in unexpected_keys and \
            'ema_num_updates' in unexpected_keys and \
            'ema_shadow_params' in unexpected_keys):
            print("Unexpected keys: ", unexpected_keys)
            raise ValueError

        self.decay = state_dict['ema_decay']
        self.num_updates = state_dict['ema_num_updates']
        self.shadow_params = state_dict['ema_shadow_params']

    def train(self, mode=True):
        if self.training == mode:
            print("Dont call model.train() with the same mode twice! Otherwise EMA parameters may overwrite original parameters")
            print("Current model training mode: ", self.training)
            print("Requested training mode: ", mode)
            raise ValueError

        nn.Module.train(self, mode)
        if mode:
            if len(self.collected_params) > 0:
                self.move_collected_params_to_model_params()
            else:
                print("model.train(True) called but no ema collected parameters!")
        else:
            self.move_model_params_to_collected_params()
            self.move_shadow_params_to_model_params()


@backward_rate_utils.register_model
class BackRateMLP(EMA,BackwardRate,GaussianTargetRate):

    def __init__(self,config,device,rank=None):
        EMA.__init__(self,config)
        BackwardRate.__init__(self,config,device,rank)

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

@backward_rate_utils.register_model
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

# make sure EMA inherited first, so it can override the state dict functions

@backward_rate_utils.register_model
class GaussianTargetRateImageX0PredEMA(EMA,ImageX0PredBase,GaussianTargetRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        ImageX0PredBase.__init__(self, cfg, device, rank)
        GaussianTargetRate.__init__(self,cfg, device)
        self.config = cfg
        self.init_ema()

all_backward_rates = {"BackRateConstant":BackRateConstant,
                      "BackRateMLP":BackRateMLP,
                      "GaussianTargetRateImageX0PredEMA":GaussianTargetRateImageX0PredEMA}

if __name__=="__main__":
    from graph_bridges.configs.graphs.lobster.config_base import BridgeConfig as GaussianBridgeConfig
    from graph_bridges.configs.graphs.lobster.config_mlp import BridgeMLPConfig

    from graph_bridges.data.dataloaders_utils import create_dataloader
    from graph_bridges.data.dataloaders import BridgeDataLoader

    # test gaussian
    gaussian_config = GaussianBridgeConfig()
    device = torch.device("cpu")

    dataloader:BridgeDataLoader
    dataloader = create_dataloader(gaussian_config,device)
    sample_ = dataloader.sample(gaussian_config.data.batch_size,device)

    gaussian_model : GaussianTargetRateImageX0PredEMA
    gaussian_model = GaussianTargetRateImageX0PredEMA(gaussian_config,device)

    time = torch.full((gaussian_config.data.batch_size,),
                      gaussian_config.sampler.min_t)
    forward = gaussian_model(sample_,time)
    print(sample_)
    print(forward.mean())

    #test mlp
    mlp_config = BridgeMLPConfig()
    device = torch.device("cpu")

    dataloader:BridgeDataLoader
    dataloader = create_dataloader(mlp_config,device)
    sample_ = dataloader.sample(mlp_config.data.batch_size,device)

    mlp_model = BackRateMLP(config=mlp_config,device=device)
    time = torch.full((mlp_config.data.batch_size,),
                      mlp_config.sampler.min_t)
    forward = mlp_model(sample_,time)
    print(sample_.shape)
    print(forward.mean())