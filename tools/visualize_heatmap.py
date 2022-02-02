# -*- coding: utf-8 -*-
"""
Created on 2019/8/4 上午9:53

@author: mick.yi

入口类

"""
import argparse
import os
import re
from glob import glob

import cv2
import numpy as np
import torch
from skimage import io
from torch import nn

from models.create_models import create_model
from utils.general import load_weight
from utils.grad_cam import GradCAM, GradCamPlusPlus
from utils.guided_back_propagation import GuidedBackPropagation


def get_net(model_prefix, model_suffix, num_classes, weight_path):

    model_name = model_prefix + '_' + model_suffix
    net = create_model(model_name=model_name, num_classes=num_classes)
    net = load_weight(net, weight_path)
    return net


def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            print(name)
            layer_name = name
    return layer_name


def prepare_input(image, means, stds):
    image = image.copy()

    # 归一化
    means = means
    stds = stds
    image -= means
    image /= stds

    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))  # channel first
    image = image[np.newaxis, ...]  # 增加batch维

    return torch.tensor(image, requires_grad=True)


def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)
    return norm_image(cam), (heatmap * 255).astype(np.uint8)


def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_gb(grad):
    """
    生guided back propagation 输入图像的梯度
    :param grad: tensor,[3,H,W]
    :return:
    """
    # 标准化
    grad = grad.data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb


def save_image(image_dicts, w, h, input_image_name, network, output_dir):
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        os.makedirs(os.path.join(output_dir, prefix, network), exist_ok=True)
        image = cv2.resize(image, (h, w), interpolation=cv2.INTER_LINEAR)
        io.imsave(os.path.join(output_dir, '{}/{}/{}.jpg'.format(prefix, network, key)), image)


def main(args):
    # 输入

    img = io.imread(args.image_path)
    w, h, _ = img.shape
    img = np.float32(cv2.resize(img, args.image_size)) / 255
    # print(img.shape)
    inputs = prepare_input(img, args.image_mean, args.image_std)
    # print(inputs.shape)
    # 输出图像
    image_dict = {}
    # 网络
    net = get_net(args.model_prefix, args.model_suffix,
                  args.num_classes, args.weight_path)
    # Grad-CAM
    layer_name = get_last_conv_name(net) if args.layer_name is None else args.layer_name
    grad_cam = GradCAM(net, layer_name, args.image_size)
    mask = grad_cam(inputs, args.class_id)  # cam mask
    # print(mask.shape)
    image_dict['cam'], image_dict['heatmap'] = gen_cam(img, mask)
    grad_cam.remove_handlers()
    # Grad-CAM++
    grad_cam_plus_plus = GradCamPlusPlus(net, layer_name, args.image_size)
    mask_plus_plus = grad_cam_plus_plus(inputs, args.class_id)  # cam mask
    image_dict['cam++'], image_dict['heatmap++'] = gen_cam(img, mask_plus_plus)
    grad_cam_plus_plus.remove_handlers()

    # GuidedBackPropagation
    gbp = GuidedBackPropagation(net)
    inputs.grad.zero_()  # 梯度置零
    grad = gbp(inputs)

    gb = gen_gb(grad)
    image_dict['gb'] = norm_image(gb)
    # 生成Guided Grad-CAM
    cam_gb = gb * mask[..., np.newaxis]
    image_dict['cam_gb'] = norm_image(cam_gb)

    network = args.model_prefix + '_' + args.model_suffix
    save_image(image_dict, w, h, os.path.basename(args.image_path), network, args.output_dir)


if __name__ == '__main__':

    #  --------------------------------------------------------------------------------------
    # |model_prefix    |model_suffix                                                         |
    # |--------------------------------------------------------------------------------------|
    # |vgg             |11 13 16 19 bn11 bn13 bn16 bn19                                      |
    # |--------------------------------------------------------------------------------------|
    # |resnet          |18 34 50 101 152                                                     |
    # |--------------------------------------------------------------------------------------|
    # |resnext         |50-32x4d 101-32x8d                                                   |
    # |--------------------------------------------------------------------------------------|
    # |regnetx         |200mf 400mf 600mf 800mf 1.6gf 3.2gf 4.0gf 6.4gf 8.0gf 12gf 16gf 32gf |
    # |--------------------------------------------------------------------------------------|
    # |regnety         |200mf 400mf 600mf 800mf 1.6gf 3.2gf 4.0gf 6.4gf 8.0gf 12gf 16gf 32gf |
    # |--------------------------------------------------------------------------------------|
    # |mobilenetv2     |0.25 0.5 0.75 1.0 1.25 1.5                                           |
    # |--------------------------------------------------------------------------------------|
    # |mobilenetv3     |small large                                                          |
    # |--------------------------------------------------------------------------------------|
    # |ghostnet        |0.5 1.0 1.3                                                          |
    # |--------------------------------------------------------------------------------------|
    # |efficientnetv1  |b0 b1 b2 b3 b4 b5 b6 b7                                              |
    # |--------------------------------------------------------------------------------------|
    # |efficientnetv2  |small medium large                                                   |
    # |--------------------------------------------------------------------------------------|
    # |shufflenetv2    |0.5 1.0 1.5 2.0                                                      |
    # |--------------------------------------------------------------------------------------|
    # |densenet        |121 161 169 201                                                      |
    # |--------------------------------------------------------------------------------------|
    # |xception        |299                                                                  |
    # |--------------------------------------------------------------------------------------|
    # |vit             |base-patch16 base-patch32 large-patch16 large-patch32 huge-patch14   |
    #  --------------------------------------------------------------------------------------
    # |resmlp-mixer    |12 24 36 B24                                                         |
    #  --------------------------------------------------------------------------------------
    # |vovnet          |27slim 39 57                                                         |
    #  --------------------------------------------------------------------------------------
    # |se-resnet       |18 34 50 101 152                                                     |
    #  --------------------------------------------------------------------------------------
    # |squeezenet      |1.0 1.1                                                              |
    #  --------------------------------------------------------------------------------------
    # |mnasnet         |0.5 0.75 1.0 1.3                                                     |
    #  --------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-prefix', type=str, default='densenet',
                        help='classification model-prefix')
    parser.add_argument('--model-suffix', type=str, default='201',
                        help='classification model-suffix')
    parser.add_argument('--num-classes', type=int, default=3,
                        help='classification num-classes')
    parser.add_argument('--image-path', type=str, default=r"D:\Torch_Classify\data\train\fish\02503_6.jpg",
                        help='input image path')
    parser.add_argument('--image-size', type=list, default=[128, 128],
                        help='input image size')
    parser.add_argument('--image-mean', type=list, default=[0.485, 0.456, 0.406],
                        help='input image mean')
    parser.add_argument('--image-std', type=list, default=[0.229, 0.224, 0.225],
                        help='input image std')
    parser.add_argument('--weight-path', type=str, default=r"D:\Torch_Classify\logs\densenet_201\exp1\last.pth",
                        help='weight path of the model')
    parser.add_argument('--layer-name', type=str, default=None,
                        help='convolutional layer name, default last')
    parser.add_argument('--class-id', type=int, default=None,
                        help='class id, default model predict id')
    parser.add_argument('--output-dir', type=str, default='vis_hp_results',
                        help='root of output directory to save results')
    arguments = parser.parse_args()

    main(arguments)
