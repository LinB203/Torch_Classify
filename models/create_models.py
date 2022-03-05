from .VGG import vgg
from .ResNet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d
from .RegNet import create_regnet
from .MobileNetv2 import mobilenetv2
from .MobileNetv3 import mobilenet_v3_large, mobilenet_v3_small
from .EfficientNetv1 import efficientnet_b0, efficientnet_b1, efficientnet_b2, \
    efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from .EfficientNetv2 import efficientnetv2_s, efficientnetv2_l, efficientnetv2_m
from .ShuffleNetv2 import shufflenet_v2_x1_0, shufflenet_v2_x0_5, shufflenet_v2_x1_5, shufflenet_v2_x2_0
from .DenseNet import densenet121, densenet161, densenet169, densenet201
from .ViT import vit_base_patch16_224_in21k, vit_base_patch32_224_in21k, vit_huge_patch14_224_in21k, \
    vit_large_patch16_224_in21k, vit_large_patch32_224_in21k
from .GhostNet import ghostnet_0_5, ghostnet_1_0, ghostnet_1_3
from .Xception import xception
from .ResMlp_mixer import resmlp_12, resmlp_24, resmlp_36, resmlpB_24
from .VoVNet import vovnet39, vovnet57, vovnet27_slim
from .Se_ResNet import se_resnet18, se_resnet34, se_resnet50, se_resnet101, se_resnet152
from .SqueezeNet import squeezenet1_0, squeezenet1_1
from .MnasNet import mnasnet0_5, mnasnet1_0, mnasnet0_75, mnasnet1_3
from .SwinT import swin_base_patch4_window7_224, swin_base_patch4_window12_384, swin_base_patch4_window7_224_in22k, \
    swin_base_patch4_window12_384_in22k, swin_small_patch4_window7_224, swin_tiny_patch4_window7_224, \
    swin_large_patch4_window7_224_in22k, swin_large_patch4_window12_384_in22k
from .ConvNeXt import convnext_base, convnext_tiny, convnext_small, convnext_large, convnext_xlarge
from .AdderNet import addernet50

def create_model(model_name, num_classes):
    model_prefix = model_name.split('_')[0]
    model_suffix = model_name.split('_')[-1]
    if model_prefix == 'vgg':
        model_suffix = model_name.split('_')[-1].split('bn')[-1]
        batch_norm = False
        if 'bn' in model_name:
            batch_norm = True
        if model_suffix == '11':
            model = vgg(model_name='vgg11', num_classes=num_classes, batch_norm=batch_norm, init_weights=True)
        elif model_suffix == '13':
            model = vgg(model_name='vgg13', num_classes=num_classes, batch_norm=batch_norm, init_weights=True)
        elif model_suffix == '16':
            model = vgg(model_name='vgg16', num_classes=num_classes, batch_norm=batch_norm, init_weights=True)
        elif model_suffix == '19':
            model = vgg(model_name='vgg19', num_classes=num_classes, batch_norm=batch_norm, init_weights=True)
        else:
            raise ValueError('[INFO] Unsupported model_suffix - `{}`, '
                             'Use 11, 13, 16, 19, bn11, bn13, bn16, bn19'.format(model_suffix))

    elif model_prefix == 'resnet':
        if model_suffix == '18':
            model = resnet18(num_classes=num_classes, include_top=True)
        elif model_suffix == '34':
            model = resnet34(num_classes=num_classes, include_top=True)
        elif model_suffix == '50':
            model = resnet50(num_classes=num_classes, include_top=True)
        elif model_suffix == '101':
            model = resnet101(num_classes=num_classes, include_top=True)
        elif model_suffix == '152':
            model = resnet152(num_classes=num_classes, include_top=True)
        else:
            raise ValueError('[INFO] Unsupported model_suffix - `{}`, '
                             'Use 18, 34, 50, 101, 152'.format(model_suffix))
    elif model_prefix == 'resnext':
        if model_suffix == '50-32x4d':
            model = resnext50_32x4d(num_classes=num_classes, include_top=True)
        elif model_suffix == '101-32x8d':
            model = resnext101_32x8d(num_classes=num_classes, include_top=True)
        else:
            raise ValueError('[INFO] Unsupported model_suffix - `{}`, '
                             'Use 50-32x4d, 101-32x8d'.format(model_suffix))

    elif model_prefix == 'regnetx' or model_prefix == 'regnety':
        if model_suffix in ['200mf', '400mf', '600mf', '800mf', '1.6gf', '3.2gf', '4.0gf', '6.4gf', '8.0gf', '12gf',
                            '16gf', '32gf']:
            model = create_regnet(model_name=model_name, num_classes=num_classes)
        else:
            raise ValueError('[INFO] Unsupported model_suffix - `{}`, '
                             'Use 200mf, 400mf, 600mf, 800mf, '
                             '1.6gf, 3.2gf, 4.0gf, 8.0gf, '
                             '12gf, 16gf, 32gf'.format(model_suffix))

    elif model_prefix == 'mobilenetv2':
        if model_suffix == '1.0':
            model = mobilenetv2(num_classes=num_classes)
        elif model_suffix == '0.25':
            model = mobilenetv2(num_classes=num_classes, alpha=0.25)
        elif model_suffix == '0.75':
            model = mobilenetv2(num_classes=num_classes, alpha=0.75)
        elif model_suffix == '1.5':
            model = mobilenetv2(num_classes=num_classes, alpha=1.5)
        elif model_suffix == '1.25':
            model = mobilenetv2(num_classes=num_classes, alpha=1.25)
        elif model_suffix == '0.5':
            model = mobilenetv2(num_classes=num_classes, alpha=0.5)
        else:
            raise ValueError('[INFO] Unsupported model_suffix - `{}`, '
                             'Use 0.25, 0.5, 0.75, 1.0, 1.25, 1.5'.format(model_suffix))

    elif model_prefix == 'mobilenetv3':
        if model_suffix == 'small':
            model = mobilenet_v3_small(num_classes=num_classes)
        elif model_suffix == 'large':
            model = mobilenet_v3_large(num_classes=num_classes)
        else:
            raise ValueError('[INFO] Unsupported model_suffix - `{}`, '
                             'Use small, large'.format(model_suffix))

    elif model_prefix == 'efficientnetv1':
        if model_suffix == 'b0':
            model = efficientnet_b0(num_classes=num_classes)
        elif model_suffix == 'b1':
            model = efficientnet_b1(num_classes=num_classes)
        elif model_suffix == 'b2':
            model = efficientnet_b2(num_classes=num_classes)
        elif model_suffix == 'b3':
            model = efficientnet_b3(num_classes=num_classes)
        elif model_suffix == 'b4':
            model = efficientnet_b4(num_classes=num_classes)
        elif model_suffix == 'b5':
            model = efficientnet_b5(num_classes=num_classes)
        elif model_suffix == 'b6':
            model = efficientnet_b6(num_classes=num_classes)
        elif model_suffix == 'b7':
            model = efficientnet_b7(num_classes=num_classes)
        else:
            raise ValueError('[INFO] Unsupported model_suffix - `{}`, '
                             'Use b0, b1, b2, b3, b4, '
                             'b5, b6, b7'.format(model_suffix))

    elif model_prefix == 'efficientnetv2':
        if model_suffix == 'small':
            model = efficientnetv2_s(num_classes=num_classes)
        elif model_suffix == 'medium':
            model = efficientnetv2_m(num_classes=num_classes)
        elif model_suffix == 'large':
            model = efficientnetv2_l(num_classes=num_classes)
        else:
            raise ValueError('[INFO] Unsupported model_suffix - `{}`, '
                             'Use small, medium, large'.format(model_suffix))

    elif model_prefix == 'shufflenetv2':
        if model_suffix == '0.5':
            model = shufflenet_v2_x0_5(num_classes=num_classes)
        elif model_suffix == '1.0':
            model = shufflenet_v2_x1_0(num_classes=num_classes)
        elif model_suffix == '1.5':
            model = shufflenet_v2_x1_5(num_classes=num_classes)
        elif model_suffix == '2.0':
            model = shufflenet_v2_x2_0(num_classes=num_classes)
        else:
            raise ValueError('[INFO] Unsupported model_suffix - `{}`, '
                             'Use 0.5, 1.0'.format(model_suffix))

    elif model_prefix == 'densenet':
        if model_suffix == '121':
            model = densenet121(num_classes=num_classes)
        elif model_suffix == '161':
            model = densenet161(num_classes=num_classes)
        elif model_suffix == '169':
            model = densenet169(num_classes=num_classes)
        elif model_suffix == '201':
            model = densenet201(num_classes=num_classes)
        else:
            raise ValueError('[INFO] Unsupported model_suffix - `{}`, '
                             'Use 121, 161, 169, 201'.format(model_suffix))

    elif model_prefix == 'vit':
        if model_suffix == 'base-patch16':
            model = vit_base_patch16_224_in21k(num_classes=num_classes, has_logits=False)
        elif model_suffix == 'base-patch32':
            model = vit_base_patch32_224_in21k(num_classes=num_classes, has_logits=False)
        elif model_suffix == 'large-patch16':
            model = vit_large_patch16_224_in21k(num_classes=num_classes, has_logits=False)
        elif model_suffix == 'large-patch32':
            model = vit_large_patch32_224_in21k(num_classes=num_classes, has_logits=False)
        elif model_suffix == 'huge-patch14':
            model = vit_huge_patch14_224_in21k(num_classes=num_classes, has_logits=False)
        else:
            raise ValueError('[INFO] Unsupported model_suffix - `{}`, '
                             'Use base-patch16, base-patch32, large-patch16, '
                             'large-patch32, huge-patch14'.format(model_suffix))
    elif model_prefix == 'ghostnet':
        if model_suffix == '0.5':
            model = ghostnet_0_5(num_classes=num_classes, width=0.5)
        elif model_suffix == '1.0':
            model = ghostnet_1_0(num_classes=num_classes, width=1.0)
        elif model_suffix == '1.3':
            model = ghostnet_1_3(num_classes=num_classes, width=1.3)
        else:
            raise ValueError('[INFO] Unsupported model_suffix - `{}`, '
                             'Use 0.5, 1.0, 1.3'.format(model_suffix))

    elif model_prefix == 'xception':
        if model_suffix == '299':
            model = xception(num_classes=num_classes)
        else:
            raise ValueError('[INFO] Unsupported model_suffix - `{}`, '
                             'Use 299'.format(model_suffix))

    elif model_prefix == 'resmlp-mixer':
        if model_suffix == '12':
            model = resmlp_12(num_classes=num_classes)
        elif model_suffix == '24':
            model = resmlp_24(num_classes=num_classes)
        elif model_suffix == '36':
            model = resmlp_36(num_classes=num_classes)
        elif model_suffix == 'B24':
            model = resmlpB_24(num_classes=num_classes)
        else:
            raise ValueError('[INFO] Unsupported model_suffix - `{}`, '
                             'Use 12, 24, 36, B24'.format(model_suffix))

    elif model_prefix == 'vovnet':
        if model_suffix == '27slim':
            model = vovnet27_slim(num_classes=num_classes)
        elif model_suffix == '39':
            model = vovnet39(num_classes=num_classes)
        elif model_suffix == '57':
            model = vovnet57(num_classes=num_classes)
        else:
            raise ValueError('[INFO] Unsupported model_suffix - `{}`, '
                             'Use 27slim, 39, 57'.format(model_suffix))

    elif model_prefix == 'se-resnet':
        if model_suffix == '18':
            model = se_resnet18(num_classes=num_classes)
        elif model_suffix == '34':
            model = se_resnet34(num_classes=num_classes)
        elif model_suffix == '50':
            model = se_resnet50(num_classes=num_classes)
        elif model_suffix == '101':
            model = se_resnet101(num_classes=num_classes)
        elif model_suffix == '152':
            model = se_resnet152(num_classes=num_classes)
        else:
            raise ValueError('[INFO] Unsupported model_suffix - `{}`, '
                             'Use 18, 34, 50, 101, 152'.format(model_suffix))

    elif model_prefix == 'squeezenet':
        if model_suffix == '1.0':
            model = squeezenet1_0(num_classes=num_classes)
        elif model_suffix == '1.1':
            model = squeezenet1_1(num_classes=num_classes)
        else:
            raise ValueError('[INFO] Unsupported model_suffix - `{}`, '
                             'Use 1.0 1.1'.format(model_suffix))

    elif model_prefix == 'mnasnet':
        if model_suffix == '0.5':
            model = mnasnet0_5(num_classes=num_classes)
        elif model_suffix == '0.75':
            model = mnasnet0_75(num_classes=num_classes)
        elif model_suffix == '1.0':
            model = mnasnet1_0(num_classes=num_classes)
        elif model_suffix == '1.3':
            model = mnasnet1_3(num_classes=num_classes)
        else:
            raise ValueError('[INFO] Unsupported model_suffix - `{}`, '
                             'Use 0.5 0.75 1.0 1.3'.format(model_suffix))

    elif model_prefix == 'swint':
        if model_suffix == 'base-224':
            model = swin_base_patch4_window7_224(num_classes=num_classes)
        elif model_suffix == 'base-384':
            model = swin_base_patch4_window12_384(num_classes=num_classes)
        elif model_suffix == 'small-224':
            model = swin_small_patch4_window7_224(num_classes=num_classes)
        elif model_suffix == 'tiny-224':
            model = swin_tiny_patch4_window7_224(num_classes=num_classes)
        elif model_suffix == 'large-224':
            model = swin_large_patch4_window7_224_in22k(num_classes=num_classes)
        elif model_suffix == 'large-384':
            model = swin_large_patch4_window12_384_in22k(num_classes=num_classes)
        else:
            raise ValueError('[INFO] Unsupported model_suffix - `{}`, '
                             'Use base-224 base-384 small-224 '
                             'tiny-224 large-224 large-384'.format(model_suffix))

    elif model_prefix == 'convnext':
        if model_suffix == 'tiny':
            model = convnext_tiny(num_classes=num_classes)
        elif model_suffix == 'small':
            model = convnext_small(num_classes=num_classes)
        elif model_suffix == 'base':
            model = convnext_base(num_classes=num_classes)
        elif model_suffix == 'large':
            model = convnext_large(num_classes=num_classes)
        elif model_suffix == 'xlarge':
            model = convnext_xlarge(num_classes=num_classes)
        else:
            raise ValueError('[INFO] Unsupported model_suffix - `{}`, '
                             'Use base small tiny large xlarge'.format(model_suffix))

    elif model_prefix == 'addernet':
        if model_suffix == '50':
            model = addernet50(num_classes=num_classes)
        else:
            raise ValueError('[INFO] Unsupported model_suffix - `{}`, '
                             'Use base 20 50'.format(model_suffix))

    else:
        raise ValueError('[INFO] Unsupported model_prefix - `{}`, '
                         'Use vgg, resnet, regnetx, regnety, '
                         'mobilenetv2, mobilenetv3, '
                         'efficientnetv1, efficientnetv2, '
                         'shufflenetv2, densenet, goolenet, '
                         'vit, ghostnet, resmlp_mixer, mnasnet'
                         'vovnet, se-resnet, squeezenet, swint, convnext'
                         'addernet'.format(model_prefix))
    return model
