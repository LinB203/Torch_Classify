import os.path
import glob
from train_config import *
from pathlib import Path
import re
import torch
from ruamel import yaml

def create_config(log_dir, configurations=configurations):
    from ruamel.yaml import YAML
    YAML = YAML()
    with open(os.path.join(log_dir, 'configs.yaml'), mode='w', encoding='utf-8') as file:
        YAML.dump(configurations, file)

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        configurations = yaml.load(f.read(), Loader=yaml.Loader)
    return configurations['cfg']

def load_train_weight(net, load_from, optimizer, starting_epoch):
    try:
        save_net = torch.load(load_from)
        pretrain_weights = save_net['net']
        public_keys = list(set(list(pretrain_weights.keys())) & set(list(net.state_dict().keys())))
        load_weights_dict = {k: pretrain_weights[k] for k in public_keys if
                             net.state_dict()[k].numel() == pretrain_weights[k].numel()}
        net.load_state_dict(load_weights_dict, strict=False)
        optimizer.load_state_dict(save_net['optimizer'])
        starting_epoch = save_net['epoch'] + 1
        # print('load net, optimizer, starting_epoch')
        return net, optimizer, starting_epoch
    except KeyError:
        pretrain_weights = torch.load(load_from)
        # print(pretrain_weights.keys())
        public_keys = list(set(list(pretrain_weights.keys())) & set(list(net.state_dict().keys())))
        # print(public_keys)
        load_weights_dict = {k: pretrain_weights[k] for k in public_keys if
                             net.state_dict()[k].numel() == pretrain_weights[k].numel()}
        # print(load_weights_dict)
        net.load_state_dict(load_weights_dict, strict=False)
        # print(a)
        # print('load net')
        return net, optimizer, starting_epoch

def load_predict_weight(net, load_from):
    try:
        save_net = torch.load(load_from)
        pretrain_weights = save_net['net']
        public_keys = list(set(list(pretrain_weights.keys())) & set(list(net.state_dict().keys())))
        load_weights_dict = {k: pretrain_weights[k] for k in public_keys if
                             net.state_dict()[k].numel() == pretrain_weights[k].numel()}
        net.load_state_dict(load_weights_dict, strict=False)
        return net
    except KeyError:
        pretrain_weights = torch.load(load_from)
        public_keys = list(set(list(pretrain_weights.keys())) & set(list(net.state_dict().keys())))
        load_weights_dict = {k: pretrain_weights[k] for k in public_keys if
                             net.state_dict()[k].numel() == pretrain_weights[k].numel()}
        net.load_state_dict(load_weights_dict, strict=False)
        # print('load net')
        return net

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    # if path.exists() and not exist_ok:
    suffix = path.suffix
    path = path.with_suffix('')
    dirs = glob.glob(f"{path}{sep}*")  # similar paths
    matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
    i = [int(m.groups()[0]) for m in matches if m]  # indices
    n = max(i) + 1 if i else 0  # increment number
    path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return str(path)
