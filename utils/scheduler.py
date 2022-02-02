from bisect import bisect_right
import math
import torch.optim.lr_scheduler as lr_scheduler
from copy import copy
def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]

def warmup_cosine_lr(epochs, warmup_epochs=5):
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    if warmup_epochs==0:
        return lambda epoch: max((1+math.cos(math.pi*epoch/epochs))/2, 0.0001)
    return lambda epoch: max((1+math.cos(math.pi*(epoch-warmup_epochs)/(epochs-warmup_epochs)))/2, 0.0001) \
                  if epoch>warmup_epochs else epoch/warmup_epochs

def warmup_step_lr(steps, epochs, warmup_epochs=5):
    if len(steps)!=0 and max(steps) >= epochs:
        raise ValueError('max(steps) should be <= epochs')
    steps = sorted(steps)
    if warmup_epochs == 0:
        return lambda epoch: 0.1**len([m for m in steps if m <= epoch])
    return lambda epoch: epoch / warmup_epochs if epoch <= warmup_epochs else 0.1**len([m for m in steps if m <= epoch])

def create_scheduler(scheduler_type, optimizer, epochs, steps, warmup_epochs=20):

    if scheduler_type == 'cosine_lr':
        # print('return cosine lr')
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lr(epochs, warmup_epochs))
        # return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lr_scale) + lr_scale)
    elif scheduler_type == 'step_lr':
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_step_lr(steps, epochs, warmup_epochs=warmup_epochs))
    else:
        raise ValueError('Unsupported scheduler_type - `{}`, '
                         'Use cosine_lr, step_lr'.format(scheduler_type))