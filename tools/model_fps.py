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

# create model
net = create_model(model_name=model_name, num_classes=num_classes)
model = net
model.to(device)
dummy_input = torch.randn(1, 3, img_size[0], img_size[1], dtype=torch.float).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 1000
timings = np.zeros((repetitions, 1))
# GPU-WARM-UP
for _ in range(200):
    _ = model(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = model(dummy_input)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
mean_fps = 1000. / mean_syn
print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn,
                                                                                     mean_fps=mean_fps))
print(mean_syn)
