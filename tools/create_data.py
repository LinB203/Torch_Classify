
import os
import shutil
import numpy as np
import random
from tqdm import tqdm

random.seed(666)
datasets_root = r'datasets'
data_root = '../data'
train_root = os.path.join(data_root, 'train')
val_root = os.path.join(data_root, 'val')
train_radio = 0.8
os.makedirs(train_root, exist_ok=True)
os.makedirs(val_root, exist_ok=True)

classes = os.listdir(datasets_root)
for class_ in tqdm(classes):
    jpgs = os.listdir(os.path.join(datasets_root, class_))
    indexes = np.arange(0, len(jpgs))
    train_indexes = random.sample(list(indexes), int(len(jpgs) * train_radio))
    val_indexes = list(set(indexes) - set(train_indexes))
    os.makedirs(os.path.join(train_root, class_), exist_ok=True)
    os.makedirs(os.path.join(val_root, class_), exist_ok=True)
    for train_index in train_indexes:
        shutil.copy(os.path.join(datasets_root, class_, jpgs[train_index]),
                    os.path.join(train_root, class_, jpgs[train_index]))
    for val_index in val_indexes:
        shutil.copy(os.path.join(datasets_root, class_, jpgs[val_index]),
                    os.path.join(val_root, class_ , jpgs[val_index]))

