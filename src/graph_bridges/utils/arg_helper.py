import argparse
import logging
import os
import random
import sys
import time
from pprint import pformat
import pickle

import networkx as nx
import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
# ref: https://github.com/lrjconan/LanczosNetwork
from torch.utils.data import TensorDataset, DataLoader

# Define a custom constructor for torch.device
def construct_torch_device(loader, node):
    value = loader.construct_scalar(node)
    return torch.device(value)

# Add the custom constructor to PyYAML
yaml.add_constructor('tag:yaml.org,2002:python/object/apply:torch.device', construct_torch_device)

from discrete_diffusion.models.graphs.gcn import GCN
from discrete_diffusion.models.graphs.gin import GIN
from discrete_diffusion.models.graphs.gnn import MAX_DEG_FEATURES
from discrete_diffusion.models.graphs.langevin_mc import LangevinMCSampler
from discrete_diffusion.models.graphs.edp_gnn import EdgeDensePredictionGraphScoreNetwork
from discrete_diffusion.models.graphs.score_network import ScoreNetwork, MLPScoreNetwork, ConvScore, UNetResScore

NAME_TO_CLASS = {
    'gcn': GCN,
    'gin': GIN,
}
def get_score_model(config, dev=None, **kwargs):
    if dev is None:
        dev = config.dev
    model_config = list(config.model.models.values())[0]
    in_features = config.dataset.in_feature
    feature_nums = [in_features + MAX_DEG_FEATURES] + model_config.feature_nums
    params = dict(model_config)
    params['feature_nums'] = feature_nums
    params.update(kwargs)
    if config.model.name == 'gnn':
        def gnn_model_func():
            return NAME_TO_CLASS[model_config.name](**params).to(dev)

        score_model = ScoreNetwork(gnn_module_func=gnn_model_func, feature_num=sum(feature_nums),
                                   stack_num=config.model_forward.stack_num if 'stack_num' in config.model_forward else 1).to(dev)
    elif config.model.name == 'mlp':
        score_model = MLPScoreNetwork(nef=64, max_node_number=config.dataset.max_node_num, dev=dev).to(dev)
    elif config.model.name == 'cov':
        score_model = ConvScore(nef=64, max_node_number=config.dataset.max_node_num, dev=dev).to(dev)
    elif config.model.name == 'unet':
        score_model = UNetResScore(nef=64, max_node_number=config.dataset.max_node_num, dev=dev).to(dev)
    elif config.model.name == 'edp-gnn':
        def gnn_model_func(**gnn_params):
            merged_params = params
            merged_params.update(gnn_params)
            return NAME_TO_CLASS[model_config.name](**merged_params).to(dev)

        feature_nums[0] = in_features
        score_model = EdgeDensePredictionGraphScoreNetwork(feature_num_list=feature_nums,
                                                           channel_num_list=model_config.channel_num_list,
                                                           max_node_number=config.dataset.max_node_num,
                                                           gnn_hidden_num_list=model_config.gnn_hidden_num_list,
                                                           gnn_module_func=gnn_model_func, dev=dev,
                                                           num_classes=len(config.train.sigmas)).to(dev)
    else:
        raise ValueError(f'Unknown model name {config.model_forward.name}')
    logging.info('model: ' + str(score_model))
    return score_model

def pad_adjs(ori_adj, node_number):
    a = ori_adj
    ori_len = a.shape[-1]
    if ori_len == node_number:
        return a
    if ori_len > node_number:
        raise ValueError(f'ori_len {ori_len} > node_number {node_number}')
    a = np.concatenate([a, np.zeros([ori_len, node_number - ori_len])], axis=-1)
    a = np.concatenate([a, np.zeros([node_number - ori_len, node_number])], axis=0)
    # a = np.logical_or(a, np.identity(node_number))
    return a

def load_dataset(data_dir='data', file_name=None, need_set=False):
    file_path = os.path.join(data_dir, file_name)
    feature_set = set()
    with open(file_path + '.pkl', 'rb') as f:
        graph_list = pickle.load(f)
    if need_set:
        for g in graph_list:
            assert isinstance(g, nx.Graph)
            feature_values = nx.get_node_attributes(g, 'feature').values()
            feature_set.update(map(lambda x: tuple(x), feature_values))
    with open(file_path + '.txt', 'r') as f:
        info = f.read()
    logging.info('load dataset: ' + info)
    # print('features set:', feature_set)
    return graph_list, feature_set

def parse_arguments(default_config="train.yaml"):
    parser = argparse.ArgumentParser(description="Running Experiments")
    parser.add_argument(
        '-c',
        '--config_file',
        type=str,
        default=os.path.join('config', default_config),
        required=True,
        help="Path of config file")
    parser.add_argument(
        '-l',
        '--log_level',
        type=str,
        default='INFO',
        help="Logging Level, one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    parser.add_argument('-m', '--comment', type=str,
                        default="", help="A single line comment for the experiment")
    args = parser.parse_args()

    return args


def get_config(config_file,comment=""):
    """ Construct and snapshot hyper parameters """
    config = edict(yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader))
    process_config(config, comment=comment)
    return config


def process_config(config, comment=''):
    # create hyper parameters
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config.dev = dev
    config.run_id = str(os.getpid())
    #config.folder_name = '_'.join([
    #    config.model_forward.name, config.dataset.name, comment.replace(' ', '_'),
    #    time.strftime('%b-%d-%H-%M-%S'), config.run_id
    #])

    #if 'save_dir' not in config:
    #    config.save_dir = os.path.join(config.exp_dir, config.exp_name, config.folder_name)
    #if 'model_save_dir' not in config:
    #    config.model_save_dir = os.path.join(config.save_dir, 'models')

    # snapshot hyper-parameters and code
    mkdir(config.exp_dir)
    #mkdir(config.save_dir)
    #mkdir(config.model_save_dir)

    # mkdir(config.save_dir + '/code')
    # os.system('cp ./*py ' + config.save_dir + '/code')
    # os.system('cp -r ./model ' + config.save_dir + '/code')
    # os.system('cp -r ./utils ' + config.save_dir + '/code')

    #save_name = os.path.join(config.save_dir, 'config.yaml')
    #yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)
    #print('config: \n' + pformat(config))


def edict2dict(edict_obj):
    dict_obj = {}

    for key, vals in edict_obj.items():
        if isinstance(vals, edict):
            dict_obj[key] = edict2dict(vals)
        else:
            dict_obj[key] = vals

    return dict_obj


def mkdir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def set_seed_and_logger(config, args):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.RandomState(config.seed)

    log_file = os.path.join(config.save_dir, args.log_level.lower() + ".log")
    FORMAT = args.comment + '| %(asctime)s %(message)s'
    fh = logging.FileHandler(log_file)
    fh.setLevel(args.log_level)
    logging.basicConfig(level=logging.DEBUG, format=FORMAT,
                        datefmt='%m-%d %H:%M:%S',
                        handlers=[
                            fh,
                            logging.StreamHandler(sys.stdout)
                        ])
    logging.info('EXPERIMENT BEGIN: ' + args.comment)
    logging.info('logging into %s', log_file)


def graphs_to_dataloader(config, graph_list):
    adjs_tensor, x_tensor = graphs_to_tensor(config, graph_list)
    train_ds = TensorDataset(adjs_tensor, x_tensor)
    train_dl = DataLoader(train_ds, batch_size=config.train.batch_size, shuffle=True)
    return train_dl


def graphs_to_tensor(config, graph_list):
    # Note: this function will set `config.dataset.in_feature` to `feature_len`
    adjs_list = []
    x_list = []
    feature_len = None
    for g in graph_list:
        assert isinstance(g, nx.Graph)
        feature_list = []
        node_list = []
        for v, feature in g.nodes.data('feature'):
            node_list.append(v)
            if feature is None:
                feature = [1.0]
            elif isinstance(feature, (float, int)):
                feature = [feature]
            else:
                feature = list(feature)
            feature_list.append(feature)
        feature_np = np.asarray(feature_list)
        if feature_len is None:
            feature_len = feature_np.shape[-1]
        else:
            assert feature_len == feature_np.shape[-1]

        # normalize node feature
        feature_np -= np.mean(feature_np, axis=0)
        feature_np /= np.std(feature_np, axis=0) + 1e-6

        padded_feature_np = np.concatenate([feature_np,
                                            np.zeros([config.dataset.max_node_num - feature_np.shape[0],
                                                      feature_len])],
                                           axis=0)
        x_list.append(padded_feature_np)

        adj = nx.to_numpy_matrix(g, nodelist=node_list)
        # print(config.dataset.max_node_num)
        padded_adj = pad_adjs(adj, node_number=config.dataset.max_node_num)
        adjs_list.append(padded_adj)

    config.dataset.in_feature = feature_len
    del graph_list

    adjs_np = np.asarray(adjs_list)
    del adjs_list

    x_np = np.asarray(x_list)
    del x_list

    adjs_tensor = torch.tensor(adjs_np, dtype=torch.float32)
    del adjs_np

    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    del x_np

    return adjs_tensor, x_tensor


def load_data(config, get_graph_list=False):
    graph_list, node_feature_set = load_dataset(data_dir='data', file_name=config.dataset.name)
    graph_list = graph_list[:config.dataset.dataset_size]
    test_size = int(config.test.split * len(graph_list))
    train_graph_list, test_graph_list = graph_list[test_size:], graph_list[:test_size],
    if get_graph_list:
        return train_graph_list, test_graph_list
    return graphs_to_dataloader(config, train_graph_list), \
        graphs_to_dataloader(config, test_graph_list)


def load_model(ckp, dev):
    model_config = edict(ckp['config'])
    model = get_score_model(config=model_config, dev=dev)
    model.load_state_dict(ckp['model'], strict=False)
    model.to(dev)
    model.eval()
    return model
