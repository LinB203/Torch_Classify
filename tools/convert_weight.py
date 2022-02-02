import os

import torch

weight_dir = r"D:\Torch_Classify\weights\\densenet121.pth"
# 查看旧名
dict = torch.load(weight_dir)
# print(dict)
old_names = list(dict.keys())
# print(old_names)
# 查看维度
for name in old_names:
    print(name, '\t\t', dict[name].shape)

# 修改参数名
# for old_name in old_names:
#     dict['backbone.model.'+old_name] = dict.pop(old_name)

# 删除参数
# for key in list(dict.keys()):
#     # if 'fc' in key: # classifier fc
#     if key in ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']:
#         del dict[key]
# torch.save(dict, r"../weights\vit/jx_vit_large_patch32_224_in21k.pth")



# 验证修改是否成功
# changed_deleted_dict = torch.load(r"../weights\torch_efficientnet\efficientnetb0.pth")
# for name in list(changed_deleted_dict.keys()):
#     print(name, '\t\t', changed_deleted_dict[name].shape)
