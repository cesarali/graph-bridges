from dataclasses import dataclass

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
