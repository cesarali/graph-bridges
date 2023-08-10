import torch

check_model_devices = lambda x: x.parameters().__next__().device
