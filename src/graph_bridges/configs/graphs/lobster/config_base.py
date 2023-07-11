from dataclasses import dataclass,asdict,field
from typing import List, Union, Optional, Tuple
import shutil
import time
import os
import subprocess
import json
from graph_bridges.data.dataloaders_config import GraphSpinsDataLoaderConfig



def get_git_revisions_hash():
    hashes = []
    hashes.append(subprocess.check_output(['git', 'rev-parse', 'HEAD']))
    hashes.append(subprocess.check_output(['git', 'rev-parse', 'HEAD^']))
    return hashes

#logs_dir = Path(save_dir).joinpath('tensorboard')
#logs_dir.mkdir(exist_ok=True)

@dataclass
class DataConfig:
    # doucet variables
    name : str = 'DoucetTargetData'
    root : str = "datasets_folder"
    train : bool = True
    download : bool = True
    batch_size : int = 28 # use 128 if you have enough memory or use distributed
    training_proportion :float = 0.8
    shuffle : bool = True

    # shapes and dimensions
    S :int = 2
    shape: List[int] = field(default_factory=lambda: [1, 1, 45])
    C: int = None
    H: int = None
    W: int = None
    D :int = None
    random_flips = True
    data_min_max : List[int]= field(default_factory=lambda:[0, 1]) # CHECK THIS for CIFAR 255


    # discrete diffusion variables
    type :str  = "doucet" #one of [doucet, spins]
    full_adjacency :bool = False
    preprocess_datapath :str = "lobster_graphs_upper"
    raw_datapath :str = "lobster_graphs_upper"

    #length = 500
    max_node :int = 10
    min_node :int = 10

    def __post_init__(self):
        self.C, self.H, self.W = self.shape[0], self.shape[1], self.shape[2]
        self.D = self.C * self.H * self.W

@dataclass
class TargetConfig:
    # doucet variables
    name : str = 'DoucetTargetData'
    root : str = "datasets_folder"
    train : bool = True
    download : bool = True
    S : int = 2
    batch_size :int = 28 # use 128 if you have enough memory or use distributed
    shuffle : bool = True

    shape : List[int] = field(default_factory=lambda : [1,1,45])
    C: int = field(init=False)
    H: int = field(init=False)
    W: int = field(init=False)

    D :int = field(init=False)

    random_flips : int = True

    # discrete diffusion variables
    type : str = "doucet" #one of [doucet, spins]
    full_adjacency : bool = False
    preprocess_datapath :str = "lobster_graphs_upper"
    raw_datapath :str = "lobster_graphs_upper"

    #length = 500
    max_node : int = 10
    min_node : int = 10

    def __post_init__(self):
        self.C, self.H, self.W = self.shape[0], self.shape[1], self.shape[2]
        self.D = self.C * self.H * self.W

@dataclass
class ModelConfig:
    name :str  = 'GaussianTargetRateImageX0PredEMA'

    # arquitecture variables
    ema_decay : float = 0.9999  # 0.9999
    do_ema :bool = True
    ch :int = 28
    num_res_blocks : int = 2
    num_scales :int = 4
    ch_mult : List[int]= field(default_factory=lambda:[1, 1, 1, 1])
    input_channels :int = 1
    scale_count_to_put_attn : int = 1
    data_min_max :List[int] = field(default_factory=lambda:[0, 1]) # CHECK THIS for CIFAR 255
    dropout :float= 0.1
    skip_rescale :bool = True
    time_embed_dim : int = 28
    time_scale_factor :int = 1000
    fix_logistic :bool = False

    # reference process variables
    initial_dist :str = 'gaussian'
    rate_sigma :float = 6.0
    Q_sigma :float = 512.0
    time_exponential :float = 3.
    time_base :float = 1.0

    def __post_init__(self):
        self.time_embed_dim = self.ch

@dataclass
class ReferenceProcessConfig:
    """
    Reference configuration for schrodinger bridge reference process
    """
    # reference process variables
    name:str = "GaussianTargetRate"
    initial_dist:str = 'gaussian'
    rate_sigma:float = 6.0
    Q_sigma:float = 512.0
    time_exponential:float = 3.
    time_base:float = 1.0

    def __init__(self,
                 name="GaussianTargetRate",
                 initial_dist='gaussian',
                 rate_sigma=6.0,
                 Q_sigma=512.0,
                 time_exponential=3.0,
                 time_base=1.0,
                 **kwargs):

        self.name = name
        self.initial_dist = initial_dist
        self.rate_sigma = rate_sigma
        self.Q_sigma = Q_sigma
        self.time_exponential = time_exponential
        self.time_base = time_base

@dataclass
class ParametrizedSamplerConfig:
    """
    Sampler for Parametrized Rates
    """
    name:str = 'TauLeaping' # TauLeaping or PCTauLeaping
    type:str = 'doucet'
    num_steps:int = 20
    min_t:float = 0.01
    eps_ratio:float = 1e-9
    initial_dist:str = 'gaussian'
    num_corrector_steps:int = 10
    corrector_step_size_multiplier:float = 1.5
    corrector_entry_time:float = 0.1

@dataclass
class SteinSpinEstimatorConfig:

    name : str = "SteinSpinEstimator"
    stein_epsilon :float = 1e-3
    stein_sample_size :int = 150

@dataclass
class BackwardEstimatorConfig:

    name : str = "BackwardRatioSteinEstimator"
    dimension_to_check : int = None

@dataclass
class CTDDLossConfig:
    name :str = 'GenericAux'
    eps_ratio :float = 1e-9
    nll_weight :float = 0.001
    min_time :float = 0.01
    one_forward_pass :bool = True

@dataclass
class CTDDSchedulerConfig:
    name :str = 'CTDDScheduler'

@dataclass
class CTDDPipelineConfig:
    name : str = 'CTDDPipeline'

@dataclass
class TrainerConfig:
    number_of_paths : int = 10
    number_of_sinkhorn : int = 1

    optimizer_name :str = 'AdamW'
    max_n_iters :int = 10000
    clip_grad :bool= True
    warmup :int = 50
    num_epochs :int = 200
    learning_rate :float = 2e-4

    gradient_accumulation_steps :int = 1
    lr_warmup_steps :int = 500
    save_image_epochs :int = 10
    save_model_epochs :int = 30
    save_model_global_iter :int = 1000

@dataclass
class BridgeConfig:

    from graph_bridges import results_path

    config_path : str = ""
    # different elements configurations------------------------------------------
    model : ModelConfig =  ModelConfig()
    data : DataConfig =  DataConfig() # corresponds to the distributions at start time
    target : DataConfig =  DataConfig() # corresponds to the distribution at final time
    reference : ReferenceProcessConfig =  ReferenceProcessConfig()
    sampler : ParametrizedSamplerConfig =  ParametrizedSamplerConfig()
    stein = SteinSpinEstimatorConfig()
    backward_estimator = BackwardEstimatorConfig()

    loss : CTDDLossConfig =  CTDDLossConfig()
    scheduler : CTDDSchedulerConfig =  CTDDSchedulerConfig()
    pipeline : CTDDPipelineConfig =  CTDDPipelineConfig()
    optimizer : TrainerConfig =  TrainerConfig()

    number_of_paths : int = 10
    number_of_sinkhorn : int = 1


    # files, directories and naming ---------------------------------------------
    delete :bool = False
    experiment_name :str = 'graph'
    experiment_type :str = 'lobster'
    experiment_indentifier :str  = 'testing'

    init_model_path = None

    # devices and parallelization ----------------------------------------------
    device = 'cpu'
    device_paths = 'cpu'
    distributed = False
    num_gpus = 0


        #data: DataConfig = DataConfig()  # corresponds to the distributions at start time
        #target: DataConfig = DataConfig()  # corresponds to the distribution at final time
        #reference: ReferenceProcessConfig = ReferenceProcessConfig()
        #sampler: ParametrizedSamplerConfig = ParametrizedSamplerConfig()
        # stein = SteinSpinEstimatorConfig()
        # backward_estimator = BackwardEstimatorConfig()
        #loss: CTDDLossConfig = CTDDLossConfig()
        #scheduler: CTDDSchedulerConfig = CTDDSchedulerConfig()
        #pipeline: CTDDPipelineConfig = CTDDPipelineConfig()
        #optimizer: TrainerConfig = TrainerConfig()
        #ModelConfig(kwargs["model"])

    def __post_init__(self):
        if isinstance(self.model,dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.data,dict):
            if self.data["name"] == "GraphSpinsDataLoader":
                self.data = GraphSpinsDataLoaderConfig(**self.data)
        if isinstance(self.target,dict):
            self.target = DataConfig(**self.target)  # corresponds to the distribution at final time
        if isinstance(self.reference,dict):
            self.reference = ReferenceProcessConfig(**self.reference)
        if isinstance(self.sampler,dict):
            self.sampler = ParametrizedSamplerConfig(**self.sampler)
        if isinstance(self.loss,dict):
            self.loss = CTDDLossConfig(**self.loss)
        if isinstance(self.scheduler,dict):
            self.scheduler = CTDDSchedulerConfig(**self.scheduler)
        if isinstance(self.pipeline,dict):
            self.pipeline = CTDDPipelineConfig(**self.pipeline)
        if isinstance(self.optimizer,dict):
            self.optimizer = TrainerConfig(**self.optimizer)


        self.current_git_commit = str(get_git_revisions_hash()[0])
        if self.experiment_indentifier is None:
            self.experiment_indentifier = str(int(time.time()))

        self.experiment_dir = os.path.join(self.results_path, self.experiment_name)
        self.experiment_type_dir = os.path.join(self.experiment_dir, self.experiment_type)
        self.results_dir = os.path.join(self.experiment_type_dir, self.experiment_indentifier)

        # doucet
        self.save_location = self.results_dir

    def initialize(self):
        self.create_directories()
        self.align_configurations()

    def align_configurations(self):
        # data distributions matches at the end
        self.data.batch_size = self.target.batch_size
        #model matches data

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
        if os.path.isdir(self.tensorboard_path) and self.delete:
            shutil.rmtree(self.tensorboard_path)
        if not os.path.isdir(self.tensorboard_path):
            os.makedirs(self.tensorboard_path)

        self.config_path = os.path.join(self.results_dir, "config.json")
        self.best_model_path = os.path.join(self.results_dir, "sinkhorn_{0}_checkpoint_{1}.tr")
        self.sinkhorn_plot_path = os.path.join(self.results_dir, "marginal_at_site_sinkhorn_{0}.png")

def get_config_from_file(experiment_name,experiment_type,experiment_indentifier)->BridgeConfig:
    from graph_bridges import results_path

    experiment_dir = os.path.join(results_path, experiment_name)
    experiment_type_dir = os.path.join(experiment_dir, experiment_type)
    results_dir = os.path.join(experiment_type_dir, experiment_indentifier)

    config_path = os.path.join(results_dir, "config.json")
    config_path_json = json.load(open(config_path,"r"))
    config_path_json["delete"] = False

    config = BridgeConfig(**config_path_json)

    return config


if __name__=="__main__":
    from pprint import pprint

    model_config = ModelConfig()
    data_config = DataConfig()
    bridge_config = BridgeConfig(delete=False)
    bridge_config2 = BridgeConfig(experiment_indentifier=None)

    config = get_config_from_file("graph", "lobster", "1688375653")
    print(config.model)
    print(config.data)
    print(config.target)
    print(config.pipeline)
    print(config.scheduler)
    print(config.reference)
    print(config.optimizer)
