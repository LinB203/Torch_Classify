import torch
import numpy as np
from train_config import configurations
from models.create_models import create_model

cfg = configurations['cfg']
img_size = cfg['img_size']
num_classes = cfg['num_classes']
device = torch.device(cfg['device'])
model_prefix = cfg['model_prefix']
model_suffix = cfg['model_suffix']
model_name = model_prefix + '_' + model_suffix
optimal_batch_size = cfg['batch_size']
# create model
net = create_model(model_name=model_name, num_classes=num_classes).to(device)
model = net
model.to(device)
dummy_input = torch.randn(optimal_batch_size, 3, img_size[0], img_size[1], dtype=torch.float).to(device)
repetitions = 100
total_time = 0
with torch.no_grad():
    for rep in range(repetitions):
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        _ = model(dummy_input)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) / 1000
        total_time += curr_time
Throughput = (repetitions * optimal_batch_size) / total_time
print('Final Throughput:', Throughput)
