import copy
import itertools
import logging
import os
from functools import reduce
from importlib import import_module
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

import torch
import yaml
from torch.autograd import grad
from typing_extensions import TypeVar

LOGGER = logging.getLogger(__name__)


def create_class_instance(module_name: str, class_name: str, kwargs, *args):
    """Create an instance of a given class.


    Args:
        module_name (str):  where the class is located
        class_name (str): _description_
        kwargs (dict): arguments needed for the class constructor

    Returns:
        class_name: instance of 'class_name'
    """
    module = import_module(module_name)
    clazz = getattr(module, class_name)
    if kwargs is None:
        instance = clazz(*args)
    else:
        instance = clazz(*args, **kwargs)

    return instance


def create_instance(name, params, *args):
    """Creates an instance of class given configuration.

    Args:
        name (string): of the module we want to create
        params (dict): dictionary containing information how to instantiate the class

    Returns:
        _type_: instance of a class
    """
    i_params = params[name]
    if type(i_params) is list:
        instance = [create_class_instance(p["module"], p["name"], p["args"], *args) for p in i_params]
    else:
        instance = create_class_instance(i_params["module"], i_params["name"], i_params["args"], *args)
    return instance


def get_static_method(module_name: str, class_name: str, method_name: str) -> Callable:
    """Get static method as function from class.

    Args:
        module_name (str): where the class is located
        class_name (str): name of the class where the function is located
        method_name (str): name of the static method

    Returns:
        Callable: static funciton
    """
    module = import_module(module_name)
    clazz = getattr(module, class_name)
    method = getattr(clazz, method_name)
    return method


def get_model_default_parameters(parameters: dict, data_loader) -> Callable:
    """Get default model parameters.

    Args:
        parameters (dict): model parameters
        data_loader (ADataLoader): used to train the model

    Returns:
        Callable: static funciton
    """
    module_name = parameters["module"]
    class_name = parameters["name"]
    module = import_module(module_name)
    clazz = getattr(module, class_name)
    method = getattr(clazz, "get_parameters")
    return method(data_loader)


def load_params(path: str, logger: Logger) -> dict:
    """Loads experiment parameters from json file.

    Args:
        path (str): to the json file
        logger (Logger):

    Returns:
        dict: param needed for the experiment
    """
    try:
        with open(path, "rb") as f:
            params = yaml.full_load(f)
        return params
    except Exception as e:
        logger.error(e)


def get_device(params: dict, rank: int = 0, logger: Logger = None, no_cuda: bool = False) -> torch.device:
    """Create device.

    Args:
        params (dict): params
        rank (int, optional): device number if using distribution training. Defaults to 0.
        logger (Logger, optional): _description_. Defaults to None.
        no_cuda (bool, optional): If ``True`` use `cpu`. Defaults to False.

    Returns:
        torch.device: _description_
    """
    # """

    # :param params:
    # :param logger:
    # :return: returns the device
    # """
    gpus = params.get("gpus", [])
    if len(gpus) > 0:
        if not torch.cuda.is_available() or no_cuda:
            if logger is not None:
                logger.warning("No GPU's available. Using CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:" + str(gpus[rank]))
    else:
        device = torch.device("cpu")
    return device


def is_primitive(v: Any) -> bool:
    """
    Checks if v is of primitive type.
    """
    return isinstance(v, (int, float, bool, str))


def free_params(module: Any) -> None:
    """Free parameters.

    Args:
        module (Any): unfreeze the model weights.
    """
    if type(module) is not list:
        module = [module]
    for m in module:
        for p in m.parameters_():
            p.requires_grad = True


def frozen_params(module: Any) -> None:
    """Freeze parameters during training.

    Args:
        module (Any): to freeyze the weights
    """
    if type(module) is not list:
        module = [module]
    for m in module:
        for p in m.parameters_():
            p.requires_grad = False


def sum_dictionaries(dicts: List) -> dict:
    """
    Sums the values of the common keys in dictionary.
    :param dicts: dictionaries containing numeric values
    :return: dictionary with summed values
    """

    def reducer(accumulator, element):
        for key, value in element.items():
            accumulator[key] = accumulator.get(key, 0) + value
        return accumulator

    return reduce(reducer, dicts, {})


def unpack_cv_parameters(params, prefix=None):
    cv_params = []
    for key, value in params.items():
        if isinstance(value, dict):
            if prefix is None:
                prefix = key
            else:
                prefix = ".".join([prefix, key])
            param_pool = unpack_cv_parameters(value, prefix)
            if "." in prefix:
                prefix = prefix.rsplit(".", 1)[0]
            else:
                prefix = None

            if len(param_pool) > 0:
                cv_params.extend(param_pool)
        elif isinstance(value, tuple) and len(value) != 0 and isinstance(value[0], dict):
            for ix, v in enumerate(value):
                if isinstance(v, dict):
                    if prefix is None:
                        prefix = key
                    else:
                        prefix = ".".join([prefix, key + f"#{ix}"])
                    param_pool = unpack_cv_parameters(v, prefix)
                    if "." in prefix:
                        prefix = prefix.rsplit(".", 1)[0]
                    else:
                        prefix = None
                    if len(param_pool) > 0:
                        cv_params.extend(param_pool)
        elif isinstance(value, list):
            if prefix is None:
                prefix = key
            else:
                key = ".".join([prefix, key])
            cv_params.append([(key, v) for v in value])
    return cv_params


def dict_set_nested(d, keys, value):
    node = d
    key_count = len(keys)
    key_idx = 0

    for key in keys:
        key_idx += 1

        if key_idx == key_count:
            node[key] = value
            return d
        else:
            if "#" in key:
                key, _id = key.split("#")
                if key not in node:
                    node[key] = dict()
                    node = node[key][int(_id)]
                else:
                    node = node[key][int(_id)]
            else:
                if key not in node:
                    node[key] = dict()
                    node = node[key]
                else:
                    node = node[key]


def convert_tuples_2_list(arg):
    for key, value in arg.items():
        if isinstance(value, dict):
            convert_tuples_2_list(value)
        else:
            if isinstance(value, tuple):
                arg[key] = list(value)

    return arg


def shuffle_tensor(x: torch.tensor, dim: int = 0) -> torch.tensor:
    return x[torch.randperm(x.shape[dim])]


def expand_params(params):
    """
    Expand the hyperparamers for grid search

    :param params:
    :return:
    """
    cv_params = []
    param_pool = unpack_cv_parameters(params)

    for i in list(itertools.product(*param_pool)):
        d = copy.deepcopy(params)
        name = d["name"]
        for j in i:
            dict_set_nested(d, j[0].split("."), j[1])
            name += "_" + j[0] + "_" + str(j[1])
            d["name"] = name.replace(".args.", "_")
        d = convert_tuples_2_list(d)
        cv_params.append(d)
    if not cv_params:
        return [params] * params["num_runs"]

    gs_params = []
    for p in cv_params:
        gs_params += [p] * p["num_runs"]
    return gs_params


def gumbel_softmax(pi, tau, device):
    """
    Gumbel-Softmax distribution.
    Implementation from https://github.com/ericjang/gumbel-softmax.
    pi: [B, ..., n_classes] class probs of categorical z
    tau: temperature
    Returns [B, ..., n_classes] as a one-hot vector
    """
    y = gumbel_softmax_sample(pi, tau, device)
    shape = y.size()
    _, ind = y.max(dim=-1)  # [B, ...]
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


def gumbel_softmax_sample(pi, tau, device, epsilon=1e-12):
    """
    Sample Gumbel-softmax
    """
    y = torch.log(pi + epsilon) + gumbel_sample(pi.size(), device)
    return torch.nn.functional.softmax(y / tau, dim=-1)


def gumbel_sample(shape, device, epsilon=1e-20):
    """
    Sample Gumbel(0,1)
    """
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + epsilon) + epsilon)


def create_nonlinearity(name):
    """
    Returns instance of non-linearity class (from torch.nn)
    """
    module = import_module("torch.nn")
    clazz = getattr(module, name)
    instance = clazz()

    return instance


def get_class_nonlinearity(name):
    """
    Returns non-linearity class (from torch.nn)
    """
    module = import_module("torch.nn")
    clazz = getattr(module, name)

    return clazz


def count_lines_in_file(file_path):
    def blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b:
                break
            yield b

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return sum(bl.count("\n") for bl in blocks(f))


def clip_grad_norm(parameters, optimizer: dict) -> None:
    if optimizer["grad_norm"] is not None:
        torch.nn.utils.clip_grad_norm_(parameters, optimizer["grad_norm"])


class OneHotEncoding(object):
    def __init__(self, encoding_size: int, ignore_index: int = -1):
        self.encoding_size = encoding_size
        self.ignore_index = ignore_index

    def __call__(self, indexes: torch.LongTensor) -> torch.FloatTensor:
        one_hot = torch.nn.functional.one_hot(indexes, self.encoding_size + 1)

        return one_hot[:, :, :-1].float()

    @property
    def embedding_dim(self):
        return self.encoding_size


def chunk_docs(n_docs: int, chunk_size: int):
    for i in range(0, n_docs, chunk_size):
        yield [i, min(i + chunk_size, n_docs)]


def jacobian(y, x):
    """Computes the Jacobian of f w.r.t x.

    This is according to the reverse mode autodiff rule,

    sum_i v^b_i dy^b_i / dx^b_j = sum_i x^b_j R_ji v^b_i,

    where:
    - b is the batch index from 0 to B - 1
    - i, j are the vector indices from 0 to M-1
    - v^b_i is a "test vector", which is set to 1 column-wise to obtain the correct
        column vectors out ot the above expression.

    :param y: torch.tensor of shape [B, M]
    :param x: torch.tensor of shape [B, N]
    :return: Jacobian matrix (torch.tensor) of shape [B, M, N]
    """

    B, M = y.shape
    jacobian = list()
    for i in range(M):
        v = torch.zeros_like(y)
        v[:, i] = 1.0
        dy_i_dx = grad(
            y,
            x,
            grad_outputs=v,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )  # shape [B, N]
        jacobian.append(torch.nn.utils.parameters_to_vector(dy_i_dx) if isinstance(x, list) else dy_i_dx[0])
    jacobian = torch.stack(jacobian, dim=1).requires_grad_()

    return jacobian


def jacobian_parameters(y, x):
    """Computes the Jacobian of f w.r.t x.

    This is according to the reverse mode autodiff rule,

    sum_i v^b_i dy^b_i / dx^b_j = sum_i x^b_j R_ji v^b_i,

    where:
    - b is the batch index from 0 to B - 1
    - i, j are the vector indices from 0 to M-1
    - v^b_i is a "test vector", which is set to 1 column-wise to obtain the correct
        column vectors out ot the above expression.

    :param y: torch.tensor of shape [B, M]
    :param x: torch.tensor of shape [B, N]
    :return: Jacobian matrix (torch.tensor) of shape [B, , N]
    """

    B, M = y.shape
    _jacobian = []

    for i in range(M):
        for j in range(B):
            dy_i_dx = grad(
                y[j, i],
                x,
                grad_outputs=torch.ones_like(y[j, i]),
                retain_graph=True,
                create_graph=True,
                allow_unused=False,
            )  # shape [B, N]
            _jacobian.append(torch.nn.utils.parameters_to_vector(dy_i_dx))

    _jacobian = torch.stack(_jacobian).reshape(B, M, -1).requires_grad_()

    return _jacobian


def second_order_jacobian(y, x):
    """Computes the Jacobian of f w.r.t x.

    This is according to the reverse mode autodiff rule,

    sum_i v^b_i dy^b_i / dx^b_j = sum_i x^b_j R_ji v^b_i,

    where:
    - b is the batch index from 0 to B - 1
    - i, j are the vector indices from 0 to M-1
    - v^b_i is a "test vector", which is set to 1 column-wise to obtain the correct
        column vectors out ot the above expression.

    :param y: torch.tensor of shape [B, M, N]
    :param x: torch.tensor of shape [B, P]
    :return: Jacobian matrix (torch.tensor) of shape [B, P, M, N]
    """

    B, M, N = y.shape
    jacobian = list()
    y = y.reshape(B, N * M)
    for i in range(N * M):
        v = torch.zeros_like(y)
        v[:, i] = 1.0
        dy_i_dx = grad(
            y,
            x,
            grad_outputs=v,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )  # shape [B, N]
        jacobian.append(torch.nn.utils.parameters_to_vector(dy_i_dx) if isinstance(x, list) else dy_i_dx[0])
    jacobian = torch.stack(jacobian, dim=1).reshape(B, M, N, -1).requires_grad_()
    return jacobian


def count_parameters(model):
    return sum(p.numel() for p in model.parameters_() if p.requires_grad)


def get_latest_checkpoint(save_dir: Path, best_model: bool = False) -> str:
    # TODO Refactor the code so the Path library is used.
    if not os.path.exists(save_dir):
        raise FileNotFoundError()
    if os.path.splitext(save_dir)[-1] == ".pth":
        return str(save_dir)
    if best_model and os.path.exists(os.path.join(save_dir, "best_model.pth")):
        return os.path.join(save_dir, "best_model.pth")
    checkpoints = [x for x in os.listdir(save_dir) if x.endswith(".pth")]
    if not checkpoints:
        raise FileNotFoundError(f"No .pth files in directory {save_dir}.")
    latest_checkpoint = sorted(checkpoints)[-1]
    return os.path.join(save_dir, latest_checkpoint)


def dict_of_tensors_2_dict_of_lists(arg1: dict):
    result = dict()
    for k, v in arg1.items():
        result[k] = v.detach().cpu().numpy().tolist()
    return result


def dict_of_tensors_2_dict_of_numbers(arg1: dict):
    result = dict()
    for k, v in arg1.items():
        result[k] = v.detach().cpu().item()
    return result


def load_model_parameters(path: str):
    model_path = get_latest_checkpoint(path, best_model=True)
    model_state = torch.load(model_path, map_location=torch.device("cpu"))
    params = model_state["params"]
    return params


def load_trained_model(path: str, device: torch.device):
    # device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    model_path = get_latest_checkpoint(path, best_model=True)
    model = load_model(model_path, device, LOGGER)
    model.eval()
    return model


def load_model(model_path: str, device: torch.device, _logger):
    model_state = torch.load(model_path, map_location=device)
    params = model_state["params"]
    _logger.info("Name of the Experiment: " + params["name"])
    if device is None:
        device = get_device(params)

    model = create_instance("model", params)
    model.load_state_dict(model_state["model_state"])
    model.to(device)
    _logger.info(model)
    return model


def load_training_dataloader(
    model_path,
    root_dir: Optional[Path] = None,
    batch_size: int = 8,
    device: Optional[torch.DeviceObjType] = None,
    return_image_path: bool = False,
):
    model_state = torch.load(model_path, map_location="cpu")
    params = model_state["params"]
    params["data_loader"]["args"]["root_dir"] = root_dir
    params["data_loader"]["args"]["validation_batch_size"] = batch_size
    params["data_loader"]["args"]["batch_size"] = batch_size
    params["data_loader"]["args"]["return_image_path"] = return_image_path

    if params["data_loader"]["args"].get("train_transform", None) is None:
        # FIXME this should be removed once we use trained models with having train transform
        params["data_loader"]["args"]["train_transform"] = {}
        params["data_loader"]["args"]["validation_transform"] = {}
        # params["data_loader"]["args"]["train_transform"]["transformations"] = {}
        # params["data_loader"]["args"]["validation_transform"]["transformations"] = {}
        params["data_loader"]["args"]["train_transform"]["transformations"] = params["data_loader"]["args"]["transform"]
        params["data_loader"]["args"]["validation_transform"]["transformations"] = params["data_loader"]["args"]["transform"]
        del params["data_loader"]["args"]["transform"]
        del params["data_loader"]["args"]["target_transform"]
    loader = create_instance("data_loader", params, device)

    return loader


def iterable_to_str(iterable: Iterable) -> str:
    return "'" + "', '".join([str(item) for item in iterable]) + "'"


T = TypeVar("T", str, bytes)


def verify_str_arg(value: T, arg: Optional[str] = None, valid_values: Optional[Iterable[T]] = None, custom_msg: Optional[str] = None) -> T:
    if not isinstance(value, torch._six.string_classes):
        if arg is None:
            msg = "Expected type str, but got type {type}."
        else:
            msg = "Expected type str for argument {arg}, but got type {type}."
        msg = msg.format(type=type(value), arg=arg)
        raise ValueError(msg)

    if valid_values is None:
        return value

    if value not in valid_values:
        if custom_msg is not None:
            msg = custom_msg
        else:
            msg = "Unknown value '{value}' for argument {arg}. Valid values are {{{valid_values}}}."
            msg = msg.format(value=value, arg=arg, valid_values=iterable_to_str(valid_values))
        raise ValueError(msg)

    return value
