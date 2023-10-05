from dataclasses import dataclass

@dataclass
class VAETrainerConfig:
    name:str = "VAETrainer"
    learning_rate: float = 1e-3
    number_of_epochs: int = 10
    save_model_epochs: int = 5

    loss_type:str = "vae_loss"
    experiment_class: str = "mnist"
    device:str = "cuda:0"
