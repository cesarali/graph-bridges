from .ctdd_reference import GaussianTargetRate
from .glauber_reference import GlauberDynamics


def load_reference(config, device):
  if config.reference.name == "GaussianTargetRate":
    image_network = GaussianTargetRate(config,device)
  elif config.reference.name == "GlauberDynamics":
    image_network = GlauberDynamics(config,device)
  else:
    raise Exception("No Reference Defined")

  return image_network
