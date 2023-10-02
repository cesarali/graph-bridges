import os
import torch
import unittest
import numpy as np
import pandas as pd
import networkx as nx
from pprint import pprint
from dataclasses import asdict

from graph_bridges.models.generative_models.sb import SB
from graph_bridges.utils.test_utils import check_model_devices

from graph_bridges.data.graph_dataloaders_config import EgoConfig
from graph_bridges.data.graph_dataloaders_config import CommunityConfig
from graph_bridges.data.graph_dataloaders_config import CommunitySmallConfig
from graph_bridges.data.graph_dataloaders_config import GridConfig


from graph_bridges.configs.config_sb import ParametrizedSamplerConfig, SteinSpinEstimatorConfig
from graph_bridges.data.spin_glass_dataloaders_config import ParametrizedSpinGlassHamiltonianConfig
from graph_bridges.data.dataloaders_utils import load_dataloader

from graph_bridges.models.temporal_networks.mlp.temporal_mlp import TemporalMLPConfig
from graph_bridges.models.temporal_networks.convnets.autoencoder import ConvNetAutoencoderConfig
from graph_bridges.models.temporal_networks.transformers.temporal_hollow_transformers import TemporalHollowTransformerConfig

class TestSB(unittest.TestCase):
    """
    Test the Schrödinger Bridge Generative Model with a super basic MLP as

    backward rate model
    """
    sb: SB

    def setUp(self) -> None:
        from graph_bridges.configs.graphs.graph_config_sb import SBConfig

        self.batch_size = 12
        self.num_time_steps = 4
        self.sb_config = SBConfig(experiment_indentifier="sb_unittest")

        #self.sb_config.data = EgoConfig(as_image=False, batch_size=self.batch_size, full_adjacency=False)

        self.sb_config.data = ParametrizedSpinGlassHamiltonianConfig(data="bernoulli_probability_0.2")
        self.sb_config.target = ParametrizedSpinGlassHamiltonianConfig(data="bernoulli_probability_0.9")

        self.sb_config.flip_estimator = SteinSpinEstimatorConfig(stein_sample_size=5)
        self.sb_config.sampler = ParametrizedSamplerConfig(num_steps=self.num_time_steps, step_type="TauLeaping")

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.sb = SB()
        self.sb.create_new_from_config(self.sb_config, self.device)

        databatch = next(self.sb.data_dataloader.train().__iter__())
        self.x_ajd = databatch[0].to(self.device)
        #self.x_features = databatch[1].to(self.device)

    #=============================================
    #devices
    #=============================================
    def test_pipeline_device(self):
        x_end,time = self.sb.pipeline(None, 0, self.device, return_path=True)
        self.assertTrue(self.device == x_end.device)

        x_end,time = self.sb.pipeline(self.sb.training_model, 1, self.device, return_path=True)
        self.assertTrue(self.device == x_end.device)

        # paths 1
        for spins_path_1, times_1 in self.sb.pipeline.paths_iterator(None,
                                                                     sinkhorn_iteration=0,
                                                                     return_path_shape=True,
                                                                     device=self.device):
            break
        self.assertTrue(self.device == spins_path_1.device)

        # paths 2
        for spins_path_2, times_2 in self.sb.pipeline.paths_iterator(self.sb.training_model,
                                                                     sinkhorn_iteration=1,
                                                                     return_path_shape=True,
                                                                     device=self.device):
            break
        self.assertTrue(self.device == spins_path_2.device)

    #=============================================
    # WITH REFERENCE PROCESS
    #=============================================

    def test_pipeline_reference_no_path(self):
        sample_size = 50
        x_end = self.sb.pipeline(None, 0, self.device, sample_size=sample_size, return_path=False)
        self.assertTrue(x_end.shape[0] == sample_size)
        self.assertIsInstance(x_end,torch.Tensor)

    def test_pipeline_reference_with_path(self):
        x_end, times = self.sb.pipeline(None, 0, self.device, return_path=True)
        self.assertIsInstance(x_end,torch.Tensor)
        self.assertIsInstance(times,torch.Tensor)
        self.assertTrue(len(x_end.shape) == 2)
        self.assertTrue(len(times.shape) == 1)

    def test_pipeline_reference_with_pathshape(self):
        x_end, times = self.sb.pipeline(None, 0, self.device, return_path=True,return_path_shape=True)
        self.assertIsInstance(x_end,torch.Tensor)
        self.assertIsInstance(times,torch.Tensor)
        self.assertTrue(len(x_end.shape) == 3)
        self.assertTrue(len(times.shape) == 2)

    def test_pipeline_reference_with_path_with_start(self):
        x_end, times = self.sb.pipeline(None, 0, self.device, self.x_ajd, return_path=True)
        self.assertIsInstance(x_end,torch.Tensor)
        self.assertIsInstance(times,torch.Tensor)

    # =============================================
    # WITH PARAMETRIC BACKWARD RATE PROCESS
    # =============================================

    def test_pipeline_parametric_backward_rate_with_path_with_start(self):
        x_end1, times1 = self.sb.pipeline(self.sb.past_model, 1, self.device, self.x_ajd, return_path=True,return_path_shape=True)
        print(x_end1.shape)

        print(times1)
        x_end2, times2 = self.sb.pipeline(self.sb.training_model, 2, self.device, self.x_ajd, return_path=True,return_path_shape=True)
        print(x_end2.shape)
        print(times2)

    def test_paths_generator(self):
        number_of_states_2 = 0
        for spins_path_2, times_2 in self.sb.pipeline.paths_iterator(self.sb.training_model,
                                                                     device=self.device,
                                                                     sinkhorn_iteration=1,
                                                                     return_path_shape=True,
                                                                     respect_batch_from_path=False):
            number_of_states_2 += spins_path_2.shape[0]
        self.assertTrue(number_of_states_2,self.sb_config.data.training_size*(self.num_time_steps+1))

        number_of_states_1 = 0
        for spins_path_1, times_1 in self.sb.pipeline.paths_iterator(None,
                                                                     sinkhorn_iteration=0,
                                                                     device=self.device,
                                                                     return_path_shape=True,
                                                                     respect_batch_from_path=False):
            number_of_states_1 += spins_path_1.shape[0]
        self.assertTrue(number_of_states_1,number_of_states_2)

    def test_balanced_shape_pipelines(self):
        from graph_bridges.configs.graphs.graph_config_sb import SBConfig
        from graph_bridges.data.dataloaders_utils import load_dataloader,check_sizes

        self.batch_size = 12
        self.num_time_steps = 8

        #================
        sb_config = SBConfig(experiment_indentifier="sb_unittest")
        sb_config.data = CommunityConfig(as_image=False, batch_size=self.batch_size, full_adjacency=True)
        sb_config.temp_network = ConvNetAutoencoderConfig()
        data_sizes = check_sizes(sb_config)
        sb_config.target = ParametrizedSpinGlassHamiltonianConfig(number_of_paths=data_sizes.total_data_size,
                                                                  number_of_spins=data_sizes.D,
                                                                  data="graph_bernoulli",
                                                                  delete_data=True,
                                                                  bernoulli_spins=True,
                                                                  bernoulli_probability=0.4)

        self.sb_config.flip_estimator = SteinSpinEstimatorConfig(stein_sample_size=5)
        self.sb_config.sampler = ParametrizedSamplerConfig(num_steps=self.num_time_steps, step_type="TauLeaping")

        self.sb = SB()
        self.sb.create_new_from_config(sb_config, self.device)

        #==========================================
        # PIPELINES
        #==========================================
        number_of_states_2 = 0
        for spins_path_2, times_2 in self.sb.pipeline.paths_iterator(self.sb.training_model,
                                                                     device=self.device,
                                                                     sinkhorn_iteration=1,
                                                                     return_path_shape=True,
                                                                     respect_batch_from_path=False):
            number_of_states_2 += spins_path_2.shape[0]
        self.assertTrue(number_of_states_2,self.sb_config.data.training_size*(self.num_time_steps+1))

        number_of_states_1 = 0
        for spins_path_1, times_1 in self.sb.pipeline.paths_iterator(None,
                                                                     sinkhorn_iteration=0,
                                                                     device=self.device,
                                                                     return_path_shape=True,
                                                                     respect_batch_from_path=False):
            number_of_states_1 += spins_path_1.shape[0]
        self.assertTrue(number_of_states_1,number_of_states_2)

if __name__ == '__main__':
    unittest.main()