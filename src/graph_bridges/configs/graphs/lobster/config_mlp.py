from dataclasses import dataclass
import shutil
import time
import os
import subprocess

def get_git_revisions_hash():
    hashes = []
    hashes.append(subprocess.check_output(['git', 'rev-parse', 'HEAD']))
    hashes.append(subprocess.check_output(['git', 'rev-parse', 'HEAD^']))
    return hashes

@dataclass
class DataConfig:
    # doucet variables
    name = 'DoucetTargetData'
    root = "datasets_folder"
    train = True
    shuffle = True
    download = True

    batch_size = 28 # use 128 if you have enough memory or use distributed
    training_proportion = 0.8

    S = 2  # number of states
    shape = [1,1,45]
    C,H,W = shape[0],shape[1],shape[2]
    D = C*H*W
    random_flips = True
    data_min_max = [0, 1] # CHECK THIS for CIFAR 255

    # discrete diffusion variables
    type = "doucet" #one of [doucet, spins]
    full_adjacency = False
    preprocess_datapath = "lobster_graphs_upper"
    raw_datapath = "lobster_graphs_upper"

    #length = 500
    max_node = 10
    min_node = 10

@dataclass
class TargetConfig:
    # doucet variables
    name = 'DoucetTargetData'
    root = "datasets_folder"

    train = True
    shuffle = True
    download = True
    batch_size = 28 # use 128 if you have enough memory or use distributed

    S = 2  # number of states
    shape = [1,1,45]
    C,H,W = shape[0],shape[1],shape[2]
    D = C*H*W
    random_flips = True
    data_min_max = [0, 1] # CHECK THIS for CIFAR 255

    # discrete diffusion variables
    type = "doucet" #one of [doucet, spins]
    full_adjacency = False
    preprocess_datapath = "lobster_graphs_upper"
    raw_datapath = "lobster_graphs_upper"

    #length = 500
    max_node = 10
    min_node = 10

@dataclass
class BackRateMLPConfig:
    name = 'BackRateMLP'

    # arquitecture variables
    ema_decay = 0.9999  # 0.9999
    time_embed_dim = 9
    hidden_layer = 200

    # reference process variables
    initial_dist = 'gaussian'
    rate_sigma = 6.0
    Q_sigma = 512.0
    time_exponential = 3.
    time_base = 1.0

@dataclass
class ReferenceProcessConfig:
    """
    Reference configuration for schrodinger bridge reference process
    """
    # reference process variables
    name = "GaussianTargetRate"
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
class SteinSpinEstimatorConfig:
    name = "SteinSpinEstimator"
    stein_epsilon = 1e-3
    stein_sample_size = 150

@dataclass
class BackwardEstimatorConfig:
    name = "BackwardRatioSteinEstimator"
    dimension_to_check = None

@dataclass
class CTDDLossConfig:
    name = 'GenericAux'
    eps_ratio = 1e-9
    nll_weight = 0.001
    min_time = 0.01
    one_forward_pass = True

@dataclass
class OptimizerConfig:
    name = 'Adam'
    lr = 2e-4
    number_of_epochs = 200

@dataclass
class BridgeMLPConfig:
    from graph_bridges import results_path

    # different elements configurations------------------------------------------
    model = BackRateMLPConfig()
    data = DataConfig() # corresponds to the distributions at start time
    target = DataConfig() # corresponds to the distribution at final time
    reference = ReferenceProcessConfig()
    sampler = ParametrizedSamplerConfig()
    stein = SteinSpinEstimatorConfig()
    backward_estimator = BackwardEstimatorConfig()
    loss = CTDDLossConfig()
    optimizer = OptimizerConfig()

    # files, directories and naming ---------------------------------------------
    delete = False
    experiment_name = 'graph'
    experiment_type = 'lobster'
    experiment_indentifier = 'testing'

    current_git_commit = str(get_git_revisions_hash()[0])
    if experiment_indentifier is None:
        experiment_indentifier = str(int(time.time()))

    experiment_dir = os.path.join(results_path,experiment_name)
    experiment_type_dir = os.path.join(experiment_dir,experiment_type)
    results_dir = os.path.join(experiment_type_dir,experiment_indentifier)

    # doucet
    save_location = results_dir
    init_model_path = None

    # devices and parallelization ----------------------------------------------
    device = 'cpu'
    device_paths = 'cpu'
    distributed = False
    num_gpus = 0

    def initialize(self):
        self.create_directories()
        self.align_configurations()

    def align_configurations(self):
        # data distributions matches at the end
        self.data.batch_size = self.target.batch_size

        #model matches data
        self.model

        # model matches reference process
        self.reference.initial_dist = self.model.initial_dist
        self.reference.rate_sigma = self.model.rate_sigma
        self.reference.Q_sigma = self.model.Q_sigma
        self.reference.time_exponential = self.model.time_exponential
        self.reference.time_base = self.model.time_base

    def create_directories(self):
        if not os.path.isdir(self.results_dir):
            os.makedirs(self.results_dir)
        else:
            if self.delete:
                shutil.rmtree(self.results_dir)
                os.makedirs(self.results_dir)

        self.tensorboard_path = os.path.join(self.results_dir, "tensorboard")
        if os.path.isdir(self.tensorboard_path):
            shutil.rmtree(self.tensorboard_path)

        self.best_model_path = os.path.join(self.results_dir, "sinkhorn_{0}.tr")
        self.sinkhorn_plot_path = os.path.join(self.results_dir, "marginal_at_site_sinkhorn_{0}.png")


if __name__=="__main__":
    model_config = BackRateMLPConfig()
    data_config = DataConfig()
    bridge_config = BridgeMLPConfig()

    bridge_config.align_configurations()
