import os
import unittest
from pprint import pprint

from graph_bridges.configs.config_oops import OopsConfig
from graph_bridges.models.generative_models.oops import OOPS
from graph_bridges.data.graph_dataloaders_config import CommunitySmallConfig
from graph_bridges.data.spin_glass_dataloaders_config import ParametrizedSpinGlassHamiltonianConfig
from graph_bridges.models.trainers.contrastive_divergence_trainer import ContrastiveDivergenceTrainer

from graph_bridges import results_path

class TestOops(unittest.TestCase):

    def test_load(self):
        experiment_name="oops"
        experiment_type="mnist"
        experiment_indentifier="test"

        experiment_dir =  os.path.join(results_path,experiment_name,experiment_type,experiment_indentifier)

        oops = OOPS()
        results, metrics, device = oops.load_from_results_folder(experiment_dir=experiment_dir)
        sample = oops.pipeline()


if __name__=="__main__":
    unittest.main()