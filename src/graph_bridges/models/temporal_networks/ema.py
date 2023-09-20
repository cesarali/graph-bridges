import torch
from torch import nn

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