
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
# |swint           |base-224 base-384 small-224 tiny-224 large-224 large-384             |
#  --------------------------------------------------------------------------------------
# |convnext        |tiny small base large xlarge                                         |
#  --------------------------------------------------------------------------------------


configurations = {
    'cfg': dict(
        config_path='',
        load_from=r"C:\Torch_Classify-master\weights\before_shufflenetv2_weights\shufflenetv2_x0.5-f707e7126e.pth",  # pretrain weight of imagenet
        model_prefix='shufflenetv2',  # above model_prefix
        model_suffix='0.5',  # above model_suffix
        img_path='data/ImageNetTE',  # the parent root where your train/val data are stored, not support test data
                          # -data
                          #    -train
                          #       -class_0
                          #          -1.jpg
                          #       -class_1
                          #       -...
                          #    -val
                          #       -class_0
                          #       -class_1
                          #       -...
        log_root='logs',  # the root to log your train/val status
        exp_name='exp', # default prefix of exp_name, will save in "model/exp_name_x"
        mean=[0.485, 0.456, 0.406],  # [0.485, 0.456, 0.406] if use pretrain weight of imagenet else [0.5, 0.5, 0.5]
        std=[0.229, 0.224, 0.225],  # [0.229, 0.224, 0.225] if use pretrain weight of imagenet else [0.5, 0.5, 0.5]
        img_size=[224, 224],  # especially for efficientnetv1 b0->224, b1->240, b2->260, b3->300, b4->380, b5->456, b6->528, b7->600
                              # especially for xception 299
                              # especially for vit 224
                              # especially for resmlp-mixer 224
        num_classes=5,
        batch_size=64,
        epochs=200,
        device="cuda:0",  #  now only support single gpu or cpu, ['cuda:0', 'cpu']
        use_benchmark=True,  # use to speed up if your img_size doesn't change dynamically
        use_apex=True,  # use apex to train by mixed-precision
        num_workers=2,  # how many workers to use for data loading. 'auto' means auto-select
        warmup_epochs=15,  # linear warm up
        optimizer_type='adamw',  # support: ['sgd', 'adam', 'adamw', 'rmsprop'] more details in utils/optimizer.py
        init_lr=0.01,
        scheduler_type='cosine_lr',  # support: ['cosine_lr', 'step_lr'] more details in utils/scheduler.py
        steps=[50, 80],  # use steps if scheduler_type=='step_lr' else ignore, default mutiply 0.1 when training epoch == step
        loss_type='LabelSmoothCELoss',  # support: ['CELoss', 'LabelSmoothCELoss', 'FocalLoss'] more details in utils/loss.py
        # especially alpha/gamma for FocalLoss
        alpha=1, # When α is a list, it is the weight of each category usually using in classify
                 # When α is constant, the category weight is [α, 1-α, 1-α, ...] usually using in object detection
        gamma=2  # retainnet set it to 2

    ),
}