import os
import torch
import unittest

#data
from graph_bridges.data.graph_dataloaders_config import CommunityConfig
from graph_bridges.data.spin_glass_dataloaders_config import ParametrizedSpinGlassHamiltonianConfig

# models
from graph_bridges.models.generative_models.sb import SB
from graph_bridges.models.temporal_networks.mlp.temporal_mlp import TemporalMLPConfig
from graph_bridges.models.temporal_networks.convnets.autoencoder import ConvNetAutoencoderConfig
from graph_bridges.models.temporal_networks.transformers.temporal_hollow_transformers import TemporalHollowTransformerConfig

from graph_bridges.models.backward_rates.sb_backward_rate_config import SchrodingerBridgeBackwardRateConfig

@unittest.skip
class TestSBBackwardRateGraphs(unittest.TestCase):

    def setUp(self) -> None:
        self.device =  torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def test_with_mlp_temp(self):
        from graph_bridges.configs.graphs.graph_config_sb import SBConfig
        sb_config = SBConfig(experiment_indentifier="sb_backward_test")

        #data
        sb_config.data = CommunityConfig(batch_size=32)

        # models
        sb_config.model = SchrodingerBridgeBackwardRateConfig()
        sb_config.temp_network = TemporalMLPConfig()

        sb = SB()
        sb.create_new_from_config(sb_config,device=self.device)

        training_model = sb.training_model
        start_databatch = next(sb.data_dataloader.train().__iter__())
        end_databatch = next(sb.target_dataloader.train().__iter__())

        x_adj_start = start_databatch[0].to(self.device)
        x_adj_end = end_databatch[0].to(self.device)
        timesteps = torch.rand(x_adj_start.shape[0]).to(self.device)

        print(f"x_adj_start {x_adj_start.shape}")
        print(f"expected shape {sb_config.data.temporal_net_expected_shape}")

        transition_rates = training_model(x_adj_start,timesteps)
        print(f"transition rates shape {transition_rates.shape}")

        """
        sb.pipeline(generation_model=training_model,
                    sinkhorn_iteration=1,
                    device=self.device,
                    initial_spins=x_adj_end,
                    sample_from_reference_native=False,
                    return_path=True,
                    return_path_shape=True)

        for databatch in sb.pipeline.paths_iterator(generation_model=training_model,
                                                    sinkhorn_iteration=1,
                                                    device=self.device,
                                                    train=True,
                                                    return_path=True,
                                                    return_path_shape=True):
            print(f"paths {databatch[0].shape}")
            print(f"time {databatch[1].shape}")
            break
        """
    def test_with_convnet_temp(self):
        from graph_bridges.configs.graphs.graph_config_sb import SBConfig
        sb_config = SBConfig(experiment_indentifier="sb_backward_test")

        #data
        sb_config.data = CommunityConfig(batch_size=5)

        # models
        sb_config.model = SchrodingerBridgeBackwardRateConfig()
        sb_config.temp_network = ConvNetAutoencoderConfig()
        sb_config.flip_estimator.stein_sample_size = 2

        sb = SB()
        sb.create_new_from_config(sb_config,device=self.device)

        training_model = sb.training_model
        past_model = sb.past_model

        start_databatch = next(sb.data_dataloader.train().__iter__())
        end_databatch = next(sb.target_dataloader.train().__iter__())

        x_adj_start = start_databatch[0].to(self.device)
        x_adj_end = end_databatch[0].to(self.device)
        timesteps = torch.rand(x_adj_start.shape[0]).to(self.device)

        print(f"x_adj_start {x_adj_start.shape}")
        print(f"expected shape {sb_config.data.temporal_net_expected_shape}")

        transition_rates = training_model(x_adj_start,timesteps)
        print(f"transition rates shape {transition_rates.shape}")

        paths, timesteps = sb.pipeline(generation_model=training_model,
                                       sinkhorn_iteration=1,
                                       device=self.device,
                                       initial_spins=x_adj_end,
                                       sample_from_reference_native=False,
                                       return_path=True,
                                       return_path_shape=False)

        print(paths.shape)
        print(timesteps.shape)

        loss_ = sb.backward_ratio_estimator.__call__(training_model,
                                                     sb.reference_process,
                                                     paths,
                                                     timesteps)
        print(loss_)

        """
        for databatch in sb.pipeline.paths_iterator(generation_model=training_model,
                                                    sinkhorn_iteration=1,
                                                    device=self.device,
                                                    train=True,
                                                    return_path=True,
                                                    return_path_shape=True):
            print(f"paths {databatch[0].shape}")
            print(f"time {databatch[1].shape}")
            break
        """

@unittest.skip
class TestSBBackwardRateSpins(unittest.TestCase):

    def setUp(self) -> None:
        self.device =  torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def test_with_mlp(self):
        from graph_bridges.configs.spin_glass.spin_glass_config_sb import SBConfig
        from graph_bridges.data.spin_glass_dataloaders_config import ParametrizedSpinGlassHamiltonianConfig

        sb_config = SBConfig(experiment_indentifier="sb_backward_test")

        #data
        sb_config.data = ParametrizedSpinGlassHamiltonianConfig(batch_size=32)

        # models
        sb_config.model = SchrodingerBridgeBackwardRateConfig()
        sb_config.temp_network = TemporalMLPConfig()

        sb = SB()
        sb.create_new_from_config(sb_config,device=self.device)

        training_model = sb.training_model
        past_model = sb.past_model

        start_databatch = next(sb.data_dataloader.train().__iter__())
        end_databatch = next(sb.target_dataloader.train().__iter__())

        x_adj_start = start_databatch[0].to(self.device)
        x_adj_end = end_databatch[0].to(self.device)
        timesteps = torch.rand(x_adj_start.shape[0]).to(self.device)

        print(f"x_adj_start {x_adj_start.shape}")
        print(f"expected shape {sb_config.data.temporal_net_expected_shape}")

        transition_rates = training_model(x_adj_start,timesteps)
        print(f"transition rates shape {transition_rates.shape}")

        paths, timesteps = sb.pipeline(generation_model=training_model,
                                       sinkhorn_iteration=1,
                                       device=self.device,
                                       initial_spins=x_adj_end,
                                       sample_from_reference_native=False,
                                       return_path=True,
                                       return_path_shape=False)

        print(paths.shape)
        print(timesteps.shape)

    def test_with_hollow_transformer(self):
        from graph_bridges.configs.spin_glass.spin_glass_config_sb import SBConfig
        from graph_bridges.data.spin_glass_dataloaders_config import ParametrizedSpinGlassHamiltonianConfig

        sb_config = SBConfig(experiment_indentifier="sb_backward_test")

        # data
        sb_config.data = ParametrizedSpinGlassHamiltonianConfig(batch_size=32)

        # models
        sb_config.model = SchrodingerBridgeBackwardRateConfig()
        sb_config.temp_network = TemporalHollowTransformerConfig()

        sb = SB()
        sb.create_new_from_config(sb_config, device=self.device)

        training_model = sb.training_model
        past_model = sb.past_model

        start_databatch = next(sb.data_dataloader.train().__iter__())
        end_databatch = next(sb.target_dataloader.train().__iter__())

        x_adj_start = start_databatch[0].to(self.device)
        x_adj_end = end_databatch[0].to(self.device)
        timesteps = torch.rand(x_adj_start.shape[0]).to(self.device)

        print(f"x_adj_start {x_adj_start.shape}")
        print(f"expected shape {sb_config.data.shape_}")

        transition_rates = training_model(x_adj_start, timesteps)
        print(f"transition rates shape {transition_rates.shape}")

        paths, timesteps = sb.pipeline(generation_model=training_model,
                                       sinkhorn_iteration=1,
                                       device=self.device,
                                       initial_spins=x_adj_end,
                                       sample_from_reference_native=False,
                                       return_path=True,
                                       return_path_shape=False)
        print(paths.shape)
        print(timesteps.shape)

class TestSBBackwardRateNIST(unittest.TestCase):

    def setUp(self) -> None:
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def test_with_convnet(self):
        from graph_bridges.configs.images.nist_config_sb import SBConfig
        from graph_bridges.data.image_dataloader_config import NISTLoaderConfig
        from graph_bridges.models.temporal_networks.convnets.autoencoder import ConvNetAutoencoderConfig

        sb_config = SBConfig(experiment_indentifier="sb_backward_test")

        #data
        sb_config.data = NISTLoaderConfig(batch_size=32)

        # models
        sb_config.model = SchrodingerBridgeBackwardRateConfig()
        sb_config.temp_network = ConvNetAutoencoderConfig()

        sb = SB()
        sb.create_new_from_config(sb_config,device=self.device)

        training_model = sb.training_model
        past_model = sb.past_model

        start_databatch = next(sb.data_dataloader.train().__iter__())
        end_databatch = next(sb.target_dataloader.train().__iter__())

        x_adj_start = start_databatch[0].to(self.device)
        x_adj_end = end_databatch[0].to(self.device)
        timesteps = torch.rand(x_adj_start.shape[0]).to(self.device)

        print(f"x_adj_start {x_adj_start.shape}")
        print(f"expected shape {sb_config.data.temporal_net_expected_shape}")

        transition_rates = training_model(x_adj_start,timesteps)
        print(f"transition rates shape {transition_rates.shape}")

        paths, timesteps = sb.pipeline(generation_model=training_model,
                                       sinkhorn_iteration=1,
                                       device=self.device,
                                       initial_spins=x_adj_end,
                                       sample_from_reference_native=False,
                                       return_path=True,
                                       return_path_shape=False)
        print(paths.shape)
        print(timesteps.shape)

    def test_with_hollow_transformer(self):
        from graph_bridges.configs.images.nist_config_sb import SBConfig
        from graph_bridges.data.image_dataloader_config import NISTLoaderConfig

        sb_config = SBConfig(experiment_indentifier="sb_backward_test")

        # data
        sb_config.data = NISTLoaderConfig(batch_size=2)
        sb_config.flip_estimator.stein_sample_size = 2

        # models
        sb_config.model = SchrodingerBridgeBackwardRateConfig()
        sb_config.temp_network = TemporalHollowTransformerConfig(num_heads=2,num_layers=2)

        sb = SB()
        sb.create_new_from_config(sb_config, device=self.device)

        training_model = sb.training_model
        past_model = sb.past_model

        start_databatch = next(sb.data_dataloader.train().__iter__())
        end_databatch = next(sb.target_dataloader.train().__iter__())

        x_adj_start = start_databatch[0].to(self.device)
        x_adj_end = end_databatch[0].to(self.device)
        timesteps = torch.rand(x_adj_start.shape[0]).to(self.device)

        print(f"x_adj_start {x_adj_start.shape}")
        print(f"expected shape {sb_config.data.temporal_net_expected_shape}")

        transition_rates = training_model(x_adj_start, timesteps)
        print(f"transition rates shape {transition_rates.shape}")

        paths, timesteps = sb.pipeline(generation_model=training_model,
                                       sinkhorn_iteration=1,
                                       device=self.device,
                                       initial_spins=x_adj_end,
                                       sample_from_reference_native=False,
                                       return_path=True,
                                       return_path_shape=False)

        print(paths.shape)
        print(timesteps.shape)

if __name__=="__main__":
    unittest.main()