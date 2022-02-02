from train_config import configurations
from models.create_models import create_model
import torch

cfg = configurations['cfg']
img_size = cfg['img_size']
num_classes = cfg['num_classes']
model_prefix = cfg['model_prefix']
model_suffix = cfg['model_suffix']
model_name = model_prefix + '_' + model_suffix
optimal_batch_size = cfg['batch_size']
# create model
model = create_model(model_name=model_name, num_classes=num_classes)
print(model.state_dict().keys())
# option1
# from torchstat import stat
# stat(model, (3, img_size[0], img_size[1]))

# option2
# from thop import profile
# input = torch.randn(1, 3, img_size[0], img_size[1])
# flops, params = profile(model, inputs=(input, ))
# print("FLOPs=", str(flops/1e6) +'{}'.format("M"))
# print("params=", str(params/1e6)+'{}'.format("M"))

# option3
# from ptflops import get_model_complexity_info
# flops, params = get_model_complexity_info(model, (3, img_size[0], img_size[1]),
#                                           as_strings=True, print_per_layer_stat=True)
# print(flops, params)

# option4   it seems a bug in py3.8?
# from fvcore.nn import FlopCountAnalysis, parameter_count_table
# tensor = (torch.rand(1, 3, img_size[0], img_size[1]),)
# flops = FlopCountAnalysis(model, tensor)
# print(flops.total())
# print(parameter_count_table(model))