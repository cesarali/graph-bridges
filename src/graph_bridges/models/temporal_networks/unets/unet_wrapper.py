import torch
from torch import nn
from dataclasses import dataclass
from graph_bridges.models.temporal_networks import networks_tau
from typing import List
from dataclasses import field

@dataclass
class UnetTauConfig():
    temp_name:"str" = "UnetTau"

    num_res_blocks : int = 2
    num_scales :int = 4
    ch_mult : List[int]= field(default_factory=lambda:[1, 1, 1, 1])
    input_channels :int = 1
    scale_count_to_put_attn : int = 1
    data_min_max :List[int] = field(default_factory=lambda:[0, 1]) # CHECK THIS for CIFAR 255
    dropout :float= 0.1
    skip_rescale :bool = True
    time_embed_dim : int = 128
    time_scale_factor :int = 1000

    def __post_init__(self):
        self.ch = self.time_embed_dim

class UnetTau(nn.Module):

    def __init__(self,cfg,device):
        super().__init__()

        ch = cfg.temp_network.ch
        num_res_blocks = cfg.temp_network.num_res_blocks
        num_scales = cfg.temp_network.num_scales
        ch_mult = cfg.temp_network.ch_mult
        input_channels = cfg.temp_network.input_channels
        output_channels = cfg.temp_network.input_channels * cfg.data.S
        scale_count_to_put_attn = cfg.temp_network.scale_count_to_put_attn
        data_min_max = cfg.temp_network.data_min_max
        dropout = cfg.temp_network.dropout
        skip_rescale = cfg.temp_network.skip_rescale
        do_time_embed = True
        time_scale_factor = cfg.temp_network.time_scale_factor
        time_embed_dim = cfg.temp_network.time_embed_dim

        self.temp_net = networks_tau.UNet(
                ch, num_res_blocks, num_scales, ch_mult, input_channels,
                output_channels, scale_count_to_put_attn, data_min_max,
                dropout, skip_rescale, do_time_embed, time_scale_factor,
                time_embed_dim
        ).to(device)

    def forward(self,x,times):
        return self.temp_net(x, times)
