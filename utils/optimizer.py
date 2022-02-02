
import torch.optim as optim


def create_optimizer(optimizer_type, net, init_lr):
    if optimizer_type == 'sgd':
        return optim.SGD(net.parameters(), lr=init_lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
    elif optimizer_type == 'adam':
        return optim.Adam(net.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
    elif optimizer_type == 'adamw':
        return optim.AdamW(net.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=True)
    elif optimizer_type == 'rmsprop':
        return optim.RMSprop(net.parameters(), lr=init_lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=True)
    else:
        raise ValueError('Unsupported scheduler_type - `{}`, '
                         'Use sgd, adam, adamw, rmsprop'.format(optimizer_type))