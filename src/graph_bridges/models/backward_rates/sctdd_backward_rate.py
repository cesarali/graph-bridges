import torch
from graph_bridges.models.backward_rates.backward_rate import BackwardRate
from graph_bridges.models.temporal_networks.transformers.temporal_hollow_transformers import TemporalHollowTransformer
from graph_bridges.models.temporal_networks.ema import EMA
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F


from torchtyping import TensorType

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

        tmp_net = TemporalHollowTransformer(

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


class GaussianTargetRateImageX0PredEMA(EMA,ImageX0PredBase,GaussianTargetRate):
    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        ImageX0PredBase.__init__(self, cfg, device, rank)
        GaussianTargetRate.__init__(self,cfg, device)
        self.config = cfg
        self.init_ema()