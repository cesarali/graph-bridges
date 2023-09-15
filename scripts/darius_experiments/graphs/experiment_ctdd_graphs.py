import argparse
import random
from graph_bridges.models.trainers.ctdd_training import CTDDTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameters for ML project")
    
    # EgoConfig params
    parser.add_argument("--batch-size", type=int, default=24, help="Batch size")
    parser.add_argument("--full-adjacency", type=bool, default=True, help="Full adjacency flag")

    # TemporalHollowTransformerConfig params
    parser.add_argument("--num-heads", type=int, default=2, help="Number of transformer heads")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--ff-hidden-dim", type=int, default=512, help="Feed-forward hidden dimension")
    parser.add_argument("--time-embed-dim", type=int, default=128, help="Time embed dimension")
    parser.add_argument("--time-scale-factor", type=int, default=10, help="Time scale factor")
    
    # ParametrizedSamplerConfig params
    parser.add_argument("--num-steps", type=int, default=100, help="Number of steps for sampler")
    
    # CTDDTrainerConfig params
    parser.add_argument("--cuda", type=int, default=0, help="Device to run on cuda:0, cuda:1, etc.")
    parser.add_argument("--num-epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--save-metric-epochs", type=int, default=10, help="Save metrics every N epochs")
    parser.add_argument("--save-model-epochs", type=int, default=10, help="Save model every N epochs")
    parser.add_argument("--save-image-epochs", type=int, default=10, help="Save images every N epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    
    from graph_bridges.configs.graphs.graph_config_ctdd import CTDDConfig
    from graph_bridges.configs.config_ctdd import CTDDTrainerConfig,ParametrizedSamplerConfig

    from graph_bridges.data.graph_dataloaders_config import EgoConfig,CommunityConfig,GridConfig,CommunitySmallConfig
    from graph_bridges.data.graph_dataloaders_config import PepperMNISTDataConfig
    from graph_bridges.data.graph_dataloaders_config import CommunitySmallConfig

    from graph_bridges.models.networks.convnets.autoencoder import ConvNetAutoencoderConfig
    from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig
    from graph_bridges.models.backward_rates.ctdd_backward_rate_config import GaussianTargetRateImageX0PredEMAConfig
    from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackwardRateTemporalHollowTransformerConfig
    from graph_bridges.models.networks.transformers.temporal_hollow_transformers import TemporalHollowTransformerConfig
    
    ctdd_config = CTDDConfig(experiment_indentifier="temporal_hollow_ego_{}".format(random.randint(0,10000)),
                             experiment_name="darius_graph",
                             experiment_type="ctdd")
                             
    ctdd_config.data = EgoConfig(batch_size=args.batch_size, full_adjacency=args.full_adjacency)

    # MLP
    # ctdd_config.model = BackRateMLPConfig()
    
    # CONVNET
    #ctdd_config.model = GaussianTargetRateImageX0PredEMAConfig()
    #ctdd_config.temp_network = ConvNetAutoencoderConfig(time_embed_dim=12,time_scale_factor=10)

    # TEMPORAL HOLLOW TRANSFORMERS
    ctdd_config.model = BackwardRateTemporalHollowTransformerConfig()
    ctdd_config.temp_network = TemporalHollowTransformerConfig(
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        ff_hidden_dim=args.ff_hidden_dim,
        time_embed_dim=args.time_embed_dim,
        time_scale_factor=args.time_scale_factor
    )

    ctdd_config.sampler = ParametrizedSamplerConfig(num_steps=args.num_steps)
    ctdd_config.trainer = CTDDTrainerConfig(
        device="cuda:{}".format(args.cuda),
        num_epochs=args.num_epochs,
        save_metric_epochs=args.save_metric_epochs,
        save_model_epochs=args.save_model_epochs,
        save_image_epochs=args.save_image_epochs,
        learning_rate=args.lr,
        metrics=["graphs", "graphs_plots", "histograms"]
    )

    ctdd_trainer = CTDDTrainer(ctdd_config)
    ctdd_trainer.train_ctdd()
