import re
import torch
weight_dir = r"D:\Torch_Classify\weights\before_densenet_weights\densenet201-c1103571.pth"
# '.'s are no longer allowed in module names, but previous _DenseLayer
# has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
# They are also in the checkpoints in model_urls. This pattern is used
# to find such keys.
pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

state_dict = torch.load(weight_dir)

# num_classes = model.classifier.out_features
# load_fc = num_classes == 1000

for key in list(state_dict.keys()):

    res = pattern.match(key)
    if res:
        new_key = res.group(1) + res.group(2)
        print(res.group(1), '                    ', res.group(2), '                    ', new_key)
        state_dict[new_key] = state_dict[key]
        del state_dict[key]

torch.save(state_dict, r"../weights\densenet201.pth")