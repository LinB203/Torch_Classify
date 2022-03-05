
import os

import torch
import torch.distributed as dist


def init_distributed_mode(cfg):

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        cfg['rank'] = int(os.environ["RANK"])
        cfg['world_size'] = int(os.environ['WORLD_SIZE'])
        cfg['gpu'] = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        cfg['rank'] = int(os.environ['SLURM_PROCID'])
        cfg['gpu'] = cfg['rank'] % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        cfg['distributed'] = False
        return

    cfg['distributed'] = True

    torch.cuda.set_device(cfg['gpu'])
    # cfg['dist_backend'] = 'gloo'
    print('| distributed init (rank {}): {}'.format(
        cfg['rank'], cfg['dist_url']), flush=True)
    dist.init_process_group(backend=cfg['dist_backend'], init_method=cfg['dist_url'],
                            world_size=cfg['world_size'], rank=cfg['rank'])
    dist.barrier()
    return cfg

def cleanup():
    dist.destroy_process_group()


def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value