from graph_bridges.data.graph_dataloaders import load_data

if __name__=="__main__":
    from graph_bridges.data.graph_dataloaders_config import CommunityConfig
    from dataclasses import asdict

    data_config = CommunityConfig()
    print(asdict(data_config))
    train_loader, test_loader = load_data(data_config)

    databatch = next(train_loader.__iter__())
    features_ = databatch[0]
    adjacencies_ = databatch[1]

    print(features_.shape)
    print(adjacencies_.shape)