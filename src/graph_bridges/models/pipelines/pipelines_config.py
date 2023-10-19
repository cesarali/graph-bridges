from dataclasses import dataclass

@dataclass
class OopsPipelineConfig:
    num_gibbs_steps:int = 100
    num_samples:int = 16
