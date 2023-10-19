import unittest
from graph_bridges.configs.config_oops import OopsConfig
from graph_bridges.data.graph_dataloaders_config import CommunitySmallConfig
from graph_bridges.data.spin_glass_dataloaders_config import ParametrizedSpinGlassHamiltonianConfig
from graph_bridges.models.trainers.contrastive_divergence_trainer import ContrastiveDivergenceTrainer

class TestOopsTrainer(unittest.TestCase):

    def test_trainer(self):
        config = OopsConfig(experiment_indentifier="test3")
        config.data = CommunitySmallConfig(as_image=False,full_adjacency=False,as_spins=False)
        config.trainer.number_of_epochs = 10
        config.trainer.learning_rate = 0.001
        config.trainer.__post_init__()

        trainer = ContrastiveDivergenceTrainer(config)
        trainer.train()

if __name__=="__main__":
    unittest.main()