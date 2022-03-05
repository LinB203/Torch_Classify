
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
# |addernet        |50                                                                   |
#  --------------------------------------------------------------------------------------



configurations = {
    'cfg': dict(
        # setup
        config_path='',           # if you want to resume a config.
        log_root='./logs',        # the root to log your train/val status
        exp_name='exp',           # default prefix of exp_name, will save in "model/exp_name_x"
        only_val=False,           # val only

        # model
        model_prefix='resnet',     # above model_prefix
        model_suffix='18',         # above model_suffix
        load_from=r"",             # pretrain weight of imagenet
        resume=False,
        use_ema=False,             # use ema to train

        # data
        num_classes=5,
        img_path='./data/test',    # the parent root where your train/val data are stored, not support test data

        # transform
        mean=[0.5, 0.5, 0.5],      # [0.485, 0.456, 0.406] if use pretrain weight of imagenet else [0.5, 0.5, 0.5]
        std=[0.5, 0.5, 0.5],       # [0.229, 0.224, 0.225] if use pretrain weight of imagenet else [0.5, 0.5, 0.5]
        mixup_prob=0.1,            # float in [0.0, 1.0]
        cutmix_prob=0.1,           # float in [0.0, 1.0]
        random_erase_prob=0.1,     # float in [0.0, 1.0]
        horizontal_flip=0.5,       # float in [0.0, 1.0]
        augment='tawide',          # ['simple', 'ra', 'tawide', 'imagenet', 'cifar10', 'svhn']
        img_size=[224, 224],       # efficientnetv1 b0:224, b1:240, b2:260, b3:300, b4:380, b5:456, b6:528, b7:600, xception 299
        val_resize=[256, 256],

        # dataloader
        num_workers=8,             # workers per gpu
        batch_size=128,            # batch size per gpu
        persistent_workers=True,   # True or False
        pin_memory=True,           # True or False

        # train
        epochs=100,
        device="cuda",             # 'cuda' or 'cpu', don't change it if you use 2 or more gpus
        use_benchmark=True,        # use to speed up if your img_size doesn't change dynamically
        use_apex=True,             # use apex to train by mixed-precision

        # optimizer
        init_lr=0.1,
        optimizer_type='sgd',      # support: ['sgd', 'adam', 'adamw', 'rmsprop']
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=False,             # for sgd
        betas=[0.9, 0.999],        # for adam or adamw
        eps=1e-6,

        # scheduler
        warmup_epochs=5,           # int >= 0
        warmup_type='linear',      # support: ['linear', 'constant']
        scheduler_type='cosine_lr',  # support: ['cosine_lr', 'step_lr', 'exponential_lr']
        steps=[30, 60, 90],        # use steps if scheduler_type=='step_lr', default lr=lr*0.1 when training epoch == (step + warmup_epochs)

        # criterion
        loss_type='CELoss',
        smoothing=0.1,             # float in [0.0, 1.0] for label smooth
        clip_grad=False,           # to clip grad if loss is nan

        # distributed
        world_size=1,              # number of gpus
        gpu_ids='0',               # don't have blank in gpu_ids
        dist_backend='nccl',       # use 'gloo' in windows, 'nccl' in linux
        dist_url='env://',
        sync_bn=False,

    ),
}
# if you use 2 or more gpus. there is a example command as follows.
# python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_gpu.py
# nproc_per_node means how many gpus you have, change world_size=nproc_per_node, and gpu_ids is you want to use(eg. '0,1')

# but if you have 2 or more gpus and want to run on a single gpu, change configures of gpu_ids above.
# remember to change configures of distributed above(eg. world_size).

# if you only have 1 gpu, change device to 'cuda', world_size=1, gpu_ids='0',then just run: python train.py.
# if you want to run on cpu, change device to 'cpu' and just run: python train.py, ignore configures of distributed above.

