import torch
import unittest

from pprint import pprint


from graph_bridges.models.generative_models.sb import SB
from graph_bridges.configs.config_sb import SBTrainerConfig
from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig
from graph_bridges.data.graph_dataloaders_config import EgoConfig
from graph_bridges.configs.graphs.graph_config_sb import SBConfig
from graph_bridges.configs.config_sb import  ParametrizedSamplerConfig, SteinSpinEstimatorConfig
from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackwardRateTemporalHollowTransformerConfig
from graph_bridges.models.networks.transformers.temporal_hollow_transformers import TemporalHollowTransformerConfig
from graph_bridges.models.trainers.sb_training import SBTrainer

class TestSBTrainer(unittest.TestCase):

    sb:SB
    sb_config:SBConfig
    sb_trainer:SBTrainer

    def setUp(self) -> None:
        self.sb_config = SBConfig(delete=True,
                                  experiment_name="graph",
                                  experiment_type="sb",
                                  experiment_indentifier="unittest_sb_trainer")
        self.sb_config.data = EgoConfig(as_image=False, batch_size=2, flatten_adjacency=True,full_adjacency=True)

        self.sb_config.model = BackRateMLPConfig()
        self.sb_config.temp_network = BackRateMLPConfig()

        self.sb_config.stein = SteinSpinEstimatorConfig(stein_sample_size=10)
        self.sb_config.sampler = ParametrizedSamplerConfig(num_steps=5)
        self.sb_config.trainer = SBTrainerConfig(learning_rate=1e-3,
                                                 num_epochs=4,
                                                 save_metric_epochs=2,
                                                 save_model_epochs=2,
                                                 save_image_epochs=2,
                                                 device="cpu",
                                                 metrics=["graphs_plots","histograms"])
        self.sb_config.__post_init__()
        self.sb_trainer = SBTrainer(self.sb_config)

    def test_training(self):
        self.sb_trainer.train_schrodinger()
        sb = SB()
        sb.load_from_results_folder(experiment_name="graph",
                                    experiment_type="sb",
                                    experiment_indentifier="unittest_sb_trainer",
                                    sinkhorn_iteration_to_load=0)
        x_end = sb.pipeline(None,0,torch.device("cpu"),sample_size=32,return_path=False)
        x_adj = sb.data_dataloader.transform_to_graph(x_end)
        print("Original")
        pprint(self.sb_config.data.__dict__)
        print("Loaded")
        pprint(sb.config.data.__dict__)


    def test_sinkhorn_initialization(self):
        current_model = self.sb_trainer.sb.training_model
        past_model = self.sb_trainer.sb.reference_process
        self.sb_trainer.initialize_sinkhorn(current_model,past_model,sinkhorn_iteration=0)

    def test_metrics_login(self):
        sinkhorn_iteration = 0
        current_model = self.sb_trainer.sb.training_model
        past_model = self.sb_trainer.sb.reference_process

        self.sb_trainer.log_metrics(current_model=current_model,
                                    past_to_train_model=past_model,
                                    sinkhorn_iteration=sinkhorn_iteration,
                                    epoch=10,
                                    device=self.sb_trainer.device)

if __name__ == '__main__':
    unittest.main()
