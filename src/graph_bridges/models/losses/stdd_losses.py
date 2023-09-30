import os
from torch import nn

class CategoricalRatioMatching():

    def __init__(self, cfg,device,rank=None):
        self.cfg = cfg
        self.ratio_eps = cfg.loss.eps_ratio
        self.nll_weight = cfg.loss.nll_weight
        self.min_time = cfg.loss.min_time
        self.cross_ent = nn.CrossEntropyLoss()
        self.device = device

    def to(self,device):
        self.device = device
        return self

    def __call__(self, minibatch,x_tilde,qt0,rate,x_logits,reg_x,p0t_sig,p0t_reg,device):
        """

        :param minibatch:
        :param state:
        :param writer:
        :return:
        """
        return minibatch