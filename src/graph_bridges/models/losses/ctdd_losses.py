import torch
from torch import nn

class GenericAux():
    def __init__(self, cfg,device,rank=None):
        self.cfg = cfg
        self.ratio_eps = cfg.loss.eps_ratio
        self.nll_weight = cfg.loss.nll_weight
        self.min_time = cfg.loss.min_time
        self.one_forward_pass = cfg.loss.one_forward_pass
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

        reg_term = self.first_term_of_elbo(minibatch,reg_x,p0t_reg,qt0,rate,device)

        outer_sum_sig,x_tilde_mask = self.second_term_of_elbo(minibatch,p0t_sig,x_tilde,qt0,rate,device)

        loss, sig_mean, reg_mean = self.second_term_of_normalization(minibatch,
                                                                     x_tilde,x_tilde_mask,
                                                                     x_logits,qt0,rate,reg_term,outer_sum_sig,device)

        return loss

    def add_noise(self, minibatch, model,ts,device):
        S = self.cfg.data.S
        if len(minibatch.shape) == 4:
            B, C, H, W = minibatch.shape
            minibatch = minibatch.view(B, C * H * W)
        B, D = minibatch.shape

        qt0 = model.transition(ts)  # (B, S, S)

        rate = model.rate(ts)  # (B, S, S)

        # --------------- Sampling x_t, x_tilde --------------------

        qt0_rows_reg = qt0[
                       torch.arange(B, device=device).repeat_interleave(D),
                       minibatch.flatten().long(),
                       :
                       ]  # (B*D, S)

        x_t_cat = torch.distributions.categorical.Categorical(qt0_rows_reg)
        x_t = x_t_cat.sample().view(B, D)

        rate_vals_square = rate[
                           torch.arange(B, device=device).repeat_interleave(D),
                           x_t.long().flatten(),
                           :
                           ]  # (B*D, S)
        rate_vals_square[
            torch.arange(B * D, device=device),
            x_t.long().flatten()
        ] = 0.0  # 0 the diagonals
        rate_vals_square = rate_vals_square.view(B, D, S)
        rate_vals_square_dimsum = torch.sum(rate_vals_square, dim=2).view(B, D)
        square_dimcat = torch.distributions.categorical.Categorical(
            rate_vals_square_dimsum
        )
        square_dims = square_dimcat.sample()  # (B,) taking values in [0, D)
        rate_new_val_probs = rate_vals_square[
                             torch.arange(B, device=device),
                             square_dims,
                             :
                             ]  # (B, S)
        square_newvalcat = torch.distributions.categorical.Categorical(
            rate_new_val_probs
        )
        square_newval_samples = square_newvalcat.sample()  # (B, ) taking values in [0, S)
        x_tilde = x_t.clone()
        x_tilde[
            torch.arange(B, device=device),
            square_dims
        ] = square_newval_samples
        # x_tilde (B, D)
        return x_t, x_tilde,qt0,rate

    def first_term_of_elbo(self, minibatch,reg_x,p0t_reg,qt0,rate,device):
        S = self.cfg.data.S
        if len(minibatch.shape) == 4:
            B, C, H, W = minibatch.shape
            minibatch = minibatch.view(B, C * H * W)
        B, D = minibatch.shape
        # ---------- First term of ELBO (regularization) ---------------

        # For (B, D, S, S) first S is x_0 second S is x'

        mask_reg = torch.ones((B, D, S), device=device)
        mask_reg[
            torch.arange(B, device=device).repeat_interleave(D),
            torch.arange(D, device=device).repeat(B),
            reg_x.long().flatten()
        ] = 0.0

        qt0_numer_reg = qt0.view(B, S, S)

        qt0_denom_reg = qt0[
                        torch.arange(B, device=device).repeat_interleave(D),
                        :,
                        reg_x.long().flatten()
                        ].view(B, D, S) + self.ratio_eps

        rate_vals_reg = rate[
                        torch.arange(B, device=device).repeat_interleave(D),
                        :,
                        reg_x.long().flatten()
                        ].view(B, D, S)

        reg_tmp = (mask_reg * rate_vals_reg) @ qt0_numer_reg.transpose(1, 2)  # (B, D, S)

        reg_term = torch.sum(
            (p0t_reg / qt0_denom_reg) * reg_tmp,
            dim=(1, 2)
        )
        return reg_term

    def second_term_of_elbo(self,minibatch,p0t_sig,x_tilde,qt0,rate,device):
        S = self.cfg.data.S
        if len(minibatch.shape) == 4:
            B, C, H, W = minibatch.shape
            minibatch = minibatch.view(B, C * H * W)
        B, D = minibatch.shape

        # ----- second term of continuous ELBO (signal term) ------------

        # When we have B,D,S,S first S is x_0, second is x

        outer_qt0_numer_sig = qt0[
            torch.arange(B, device=device).repeat_interleave(D * S),
            minibatch.long().flatten().repeat_interleave(S),
            torch.arange(S, device=device).repeat(B * D)
        ].view(B, D, S)

        outer_qt0_denom_sig = qt0[
                                  torch.arange(B, device=device).repeat_interleave(D),
                                  minibatch.long().flatten(),
                                  x_tilde.long().flatten()
                              ] + self.ratio_eps  # (B, D)

        qt0_numer_sig = qt0.view(B, S, S)  # first S is x_0, second S is x

        qt0_denom_sig = qt0[
                        torch.arange(B, device=device).repeat_interleave(D),
                        :,
                        x_tilde.long().flatten()
                        ].view(B, D, S) + self.ratio_eps

        inner_log_sig = torch.log(
            (p0t_sig / qt0_denom_sig) @ qt0_numer_sig + self.ratio_eps
        )  # (B, D, S)

        x_tilde_mask = torch.ones((B, D, S), device=device)
        x_tilde_mask[
            torch.arange(B, device=device).repeat_interleave(D),
            torch.arange(D, device=device).repeat(B),
            x_tilde.long().flatten()
        ] = 0.0

        outer_rate_sig = rate[
            torch.arange(B, device=device).repeat_interleave(D * S),
            torch.arange(S, device=device).repeat(B * D),
            x_tilde.long().flatten().repeat_interleave(S)
        ].view(B, D, S)

        outer_sum_sig = torch.sum(
            x_tilde_mask * outer_rate_sig * (outer_qt0_numer_sig / outer_qt0_denom_sig.view(B, D, 1)) * inner_log_sig,
            dim=(1, 2)
        )

        # now getting the 2nd term normalization
        return outer_sum_sig,x_tilde_mask

    def second_term_of_normalization(self,minibatch,
                                     x_tilde,x_tilde_mask,
                                     x_logits,qt0,rate,reg_term,outer_sum_sig,device):
        """

        :return:
        """
        S = self.cfg.data.S
        if len(minibatch.shape) == 4:
            B, C, H, W = minibatch.shape
            minibatch = minibatch.view(B, C * H * W)
        B, D = minibatch.shape

        rate_row_sums = - rate[
            torch.arange(B, device=device).repeat_interleave(S),
            torch.arange(S, device=device).repeat(B),
            torch.arange(S, device=device).repeat(B)
        ].view(B, S)

        base_Z_tmp = rate_row_sums[
            torch.arange(B, device=device).repeat_interleave(D),
            x_tilde.long().flatten()
        ].view(B, D)
        base_Z = torch.sum(base_Z_tmp, dim=1)

        Z_subtraction = base_Z_tmp  # (B,D)
        Z_addition = rate_row_sums

        Z_sig_norm = base_Z.view(B, 1, 1) - \
                     Z_subtraction.view(B, D, 1) + \
                     Z_addition.view(B, 1, S)

        rate_sig_norm = rate[
            torch.arange(B, device=device).repeat_interleave(D * S),
            torch.arange(S, device=device).repeat(B * D),
            x_tilde.long().flatten().repeat_interleave(S)
        ].view(B, D, S)

        # qt0 is (B,S,S)
        qt0_sig_norm_numer = qt0[
            torch.arange(B, device=device).repeat_interleave(D * S),
            minibatch.long().flatten().repeat_interleave(S),
            torch.arange(S, device=device).repeat(B * D)
        ].view(B, D, S)

        qt0_sig_norm_denom = qt0[
                                 torch.arange(B, device=device).repeat_interleave(D),
                                 minibatch.long().flatten(),
                                 x_tilde.long().flatten()
                             ].view(B, D) + self.ratio_eps

        sig_norm = torch.sum(
            (rate_sig_norm * qt0_sig_norm_numer * x_tilde_mask) / (Z_sig_norm * qt0_sig_norm_denom.view(B, D, 1)),
            dim=(1, 2)
        )

        sig_mean = torch.mean(- outer_sum_sig / sig_norm)

        reg_mean = torch.mean(reg_term)

        neg_elbo = sig_mean + reg_mean

        perm_x_logits = torch.permute(x_logits, (0, 2, 1))

        nll = self.cross_ent(perm_x_logits, minibatch.long())
        loss = neg_elbo + self.nll_weight * nll

        return loss, sig_mean, reg_mean
