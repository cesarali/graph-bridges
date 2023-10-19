from graph_bridges.data.spin_glass_dataloaders_config import ParametrizedSpinGlassHamiltonianConfig

from graph_bridges.data.graph_dataloaders_config import (
    EgoConfig,
    CommunityConfig,
    CommunitySmallConfig,
    GridConfig,
    EnzymesConfig,
    QM9Config,
    ZincConfig,
    GraphSpinsDataLoaderConfig,
    TargetConfig
)

all_dataloaders_configs = {"ego_small":EgoConfig,
                           "community_small":CommunitySmallConfig,
                           "community":CommunityConfig,
                           "grid":GridConfig,
                           "ENZYMES":EnzymesConfig,
                           "QM9":QM9Config,
                           "ZINC250k":ZincConfig,
                           "ParametrizedSpinGlassHamiltonian":ParametrizedSpinGlassHamiltonianConfig,
                           "GraphSpinsDataLoader":GraphSpinsDataLoaderConfig,
                           "DoucetTargetData":TargetConfig}