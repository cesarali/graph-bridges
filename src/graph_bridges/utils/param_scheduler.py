import numpy as np


class ExponentialScheduler(object):
    max_steps: int
    decay_rate: float
    max_value: int

    def __init__(self, **kwargs):
        self.max_steps = kwargs.get("max_steps", 1000)
        self.decay_rate = kwargs.get("decay_rate", 0.1)
        self.max_value = kwargs.get("max_value", 1.0)

    def __call__(self, step):
        return float(self.max_value / (1.0 + np.exp(-self.decay_rate * (step - self.max_steps))))


class ExponentialSchedulerGumbel(object):
    def __init__(self, **kwargs):
        self.min_tau = kwargs.get("min_temp")
        self.decay_rate = kwargs.get("decay_rate")

    def __call__(self, tau_init, step):
        t = np.maximum(tau_init * np.exp(-self.decay_rate * step), self.min_tau)
        return t


class LinearScheduler(object):
    def __init__(self, **kwargs):
        self.max_steps = kwargs.get("max_steps", 1000)
        self.start_value = kwargs.get("start_value", 0)
        print("start_value linear scheduler {}".format(self.start_value))

    def __call__(self, step):
        if self.start_value == 0:
            return min(1.0, float(step) / self.max_steps)
        else:
            return min(
                1.0,
                self.start_value + float(step) / self.max_steps * (1 - self.start_value),
            )


class ConstantScheduler(object):
    def __init__(self, **kwargs):
        self.beta = kwargs.get("beta", 1000)

    def __call__(self, step):
        return self.beta
