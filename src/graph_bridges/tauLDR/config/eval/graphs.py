import ml_collections

def get_config():

    datasets_folder = 'C:/Users/cesar/Desktop/Projects/DiffusiveGenerativeModelling/Codes/discrete-diffusion/data/raw/cifar10/cifar10'
    model_location = 'C:/Users/cesar/Desktop/Projects/DiffusiveGenerativeModelling/Codes/discrete-diffusion/results/tauLDR/2023-06-02/18-59-11_graphs/checkpoints/ckpt_0000001999.pt'
    model_config_location = 'C:/Users/cesar/Desktop/Projects/DiffusiveGenerativeModelling/Codes/discrete-diffusion/results/tauLDR/2023-06-02/18-59-11_graphs/config/config_001.yaml'

    config = ml_collections.ConfigDict()
    config.eval_name = 'graphs'
    config.train_config_overrides = [
        [['device'], 'cpu'],
        [['data', 'root'], datasets_folder],
        [['distributed'], False]
    ]
    config.train_config_path = model_config_location
    config.checkpoint_path = model_location

    config.device = 'cpu'

    config.data = data = ml_collections.ConfigDict()
    data.name = 'graphs'
    data.root = datasets_folder
    data.train = True
    data.download = True
    data.S = 2
    data.batch_size = 28 # use 128 if you have enough memory or use distributed
    data.shuffle = True
    data.shape = [1,1,45]
    data.random_flips = True

    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = 'TauLeaping' # TauLeaping or PCTauLeaping
    sampler.num_steps = 1000
    sampler.min_t = 0.01
    sampler.eps_ratio = 1e-9
    sampler.initial_dist = 'gaussian'
    sampler.num_corrector_steps = 10
    sampler.corrector_step_size_multiplier = 1.5
    sampler.corrector_entry_time = 0.1

    return config
