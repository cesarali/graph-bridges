from dataclasses import dataclass

@dataclass
class RealFlipConfig:
    name: str = "RealFlip"

@dataclass
class GradientEstimatorConfig:
    name: str = "GradientEstimator"
    stein_epsilon: float = 1e-3
    stein_sample_size: int = 150

@dataclass
class SteinSpinEstimatorConfig:
    name : str = "SteinSpinEstimator"
    stein_epsilon :float = 0.2
    stein_sample_size :int = 200

@dataclass
class CTDDLossConfig:
    name :str = 'GenericAux'
    eps_ratio :float = 1e-9
    nll_weight :float = 0.001
    min_time :float = 0.01
    one_forward_pass :bool = True

@dataclass
class SCTDDLossConfig:
    name :str = 'GenericAux'
    eps_ratio :float = 1e-9
    nll_weight :float = 0.001
    min_time :float = 0.01
    one_forward_pass :bool = True


all_loss_configs = {"GradientEstimator":GradientEstimatorConfig,
                    "SteinSpinEstimator":SteinSpinEstimatorConfig,
                    "RealFlipConfig":RealFlipConfig,
                    "CTDDLoss":CTDDLossConfig,
                    "SCTDDLoss":SCTDDLossConfig}