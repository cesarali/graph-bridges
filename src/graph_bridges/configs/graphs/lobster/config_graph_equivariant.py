from dataclasses import dataclass
import os

@dataclass
class DataConfig:
    # doucet variables
    name = 'lobster'
    root = "datasets_folder"
    train = True
    download = True
    S = 2
    batch_size = 28 # use 128 if you have enough memory or use distributed
    shuffle = True

    shape = [1,1,45]
    C,H,W = shape[0],shape[1],shape[2]
    D = C*H*W
    random_flips = True

    # discrete diffusion variables
    type = "doucet" #one of [doucet, spins]
    full_adjacency = False
    preprocess_datapath = "lobster_graphs_upper"
    raw_datapath = "lobster_graphs_upper"

    #length = 500
    max_node = 10
    min_node = 10

@dataclass
class ModelConfig:
    name = 'GraphScorePerTime'

    # arquitecture variables
    time_embed_dim = 9
    channel_num_list = [2,4,4,4,2]
    dropout_p = 0.0
    feature_nums = [time_embed_dim+1,16,16,16,16,16]
    gnn_hidden_num_list = [16,16,16,16]
    graph_func_name = 'gin'
    max_node_num = 10
    number_of_classes = 1
    number_of_spins = 100
    use_norm_layers = False
    number_of_classes = 1
    #number_of_spins = paths_dataloader.number_of_spins

    # reference process variables
    initial_dist = 'gaussian'
    rate_sigma = 6.0
    Q_sigma = 512.0
    time_exponential = 3.
    time_base = 1.0

class ReferenceProcessConfig:
    """
    Reference configuration for schrodinger bridge reference process
    """
    # reference process variables
    initial_dist = 'gaussian'
    rate_sigma = 6.0
    Q_sigma = 512.0
    time_exponential = 3.
    time_base = 1.0

@dataclass
class ParametrizedSamplerConfig:
    """
    Sampler for Parametrized Rates
    """
    name = 'TauLeaping' # TauLeaping or PCTauLeaping
    type = 'doucet'
    num_steps = 1000
    min_t = 0.01
    eps_ratio = 1e-9
    initial_dist = 'gaussian'
    num_corrector_steps = 10
    corrector_step_size_multiplier = 1.5
    corrector_entry_time = 0.1

@dataclass
class BridgeConfig:

    from graph_bridges import results_path

    # different elements configurations------------------------------------------
    model = ModelConfig()
    data = DataConfig() # corresponds to the distributions at start time
    target = DataConfig() # corresponds to the distribution at final time
    reference_process = ReferenceProcessConfig()
    sampler = ParametrizedSamplerConfig()

    # files, directories and naming ---------------------------------------------
    experiment_name = 'graph'
    experiment_type = 'lobster'
    experiment_indentifier = 'testing'

    experiment_dir = os.path.join(results_path,experiment_name)
    experiment_type_dir = os.path.join(experiment_dir,experiment_type)
    results_dir = os.path.join(experiment_type_dir,experiment_indentifier)

    save_location = results_dir
    init_model_path = None

    # devices and parallelization ----------------------------------------------
    device = 'cpu'
    distributed = False
    num_gpus = 0

    def initialize(self):
        return None

    def align_configurations(self):
        # data distributions matches at the end
        self.data.batch_size = self.target.batch_size

        # model matches data

        # model matches reference process
        self.reference_process.initial_dist = self.model.initial_dist
        self.reference_process.rate_sigma = self.model.rate_sigma
        self.reference_process.Q_sigma = self.model.Q_sigma
        self.reference_process.time_exponential = self.model.time_exponential
        self.reference_process.time_base = self.model.time_base

    def create_directories(self):
        return None

if __name__=="__main__":
    model_config = ModelConfig()
    data_config = DataConfig()
    bridge_config = BridgeConfig()

    bridge_config.align_configurations()

    """
    device = torch.device("cpu")
    model = GaussianTargetRateImageX0PredEMA(bridge_config,device)
    X = torch.Tensor(size=(data_config.batch_size,45)).normal_(0.,1.)
    time = torch.Tensor(size=(data_config.batch_size,)).uniform_(0.,1.)
    forward = model(X,time)
    print(forward.shape)
    """