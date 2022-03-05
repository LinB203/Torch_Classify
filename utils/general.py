import os.path
import glob
from pathlib import Path
import re
import torch
from ruamel import yaml

def create_config(log_dir, cfg):
    from ruamel.yaml import YAML
    YAML = YAML()
    with open(os.path.join(log_dir, 'configs.yaml'), mode='w', encoding='utf-8') as file:
        YAML.dump(cfg, file)

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.Loader)
    return cfg

def load_train_weight(net, load_from, optimizer, scheduler, scaler, net_ema, cfg):
    try:
        save_net = torch.load(load_from, map_location="cpu")
        pretrain_weights = save_net['net']
        public_keys = list(set(list(pretrain_weights.keys())) & set(list(net.state_dict().keys())))
        load_weights_dict = {k: pretrain_weights[k] for k in public_keys if
                             net.state_dict()[k].numel() == pretrain_weights[k].numel()}
        net.load_state_dict(load_weights_dict, strict=False)

        if cfg['resume']:
            optimizer.load_state_dict(save_net['optimizer'])
            scheduler.load_state_dict(save_net['scheduler'])
            try:
                scaler.load_state_dict(save_net['scaler'])
            except:
                pass
            try:
                net_ema.load_state_dict(save_net['net_ema'])
            except:
                pass
        print('[INFO] Successfully Load Weight From {}...'.format(load_from))
    except KeyError:
        pretrain_weights = torch.load(load_from, map_location="cpu")
        # print(pretrain_weights.keys())
        public_keys = list(set(list(pretrain_weights.keys())) & set(list(net.state_dict().keys())))
        # print(public_keys)
        load_weights_dict = {k: pretrain_weights[k] for k in public_keys if
                             net.state_dict()[k].numel() == pretrain_weights[k].numel()}
        # print(load_weights_dict)
        net.load_state_dict(load_weights_dict, strict=False)
        print('[INFO] Successfully Load Weight From {}...'.format(load_from))

def load_predict_weight(net, load_from):
    try:
        save_net = torch.load(load_from, map_location="cpu")
        pretrain_weights = save_net['net']
        public_keys = list(set(list(pretrain_weights.keys())) & set(list(net.state_dict().keys())))
        load_weights_dict = {k: pretrain_weights[k] for k in public_keys if
                             net.state_dict()[k].numel() == pretrain_weights[k].numel()}
        net.load_state_dict(load_weights_dict, strict=False)
        # return net
    except KeyError:
        pretrain_weights = torch.load(load_from, map_location="cpu")
        public_keys = list(set(list(pretrain_weights.keys())) & set(list(net.state_dict().keys())))
        load_weights_dict = {k: pretrain_weights[k] for k in public_keys if
                             net.state_dict()[k].numel() == pretrain_weights[k].numel()}
        net.load_state_dict(load_weights_dict, strict=False)
        # print('load net')
        # return net

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



class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg)

    def update_parameters(self, model):
        for p_swa, p_model in zip(self.module.state_dict().values(), model.state_dict().values()):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(self.avg_fn(p_swa.detach(), p_model_, self.n_averaged.to(device)))
        self.n_averaged += 1