def main(cfg):
    import os
    import json
    import warnings

    warnings.filterwarnings("ignore")
    import numpy as np
    from tqdm import tqdm
    from time import time

    import torch
    from torch import nn
    from torch import optim
    from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR, LinearLR, ConstantLR, \
        SequentialLR
    from torch.utils.data import DataLoader, DistributedSampler, BatchSampler
    from tensorboardX import SummaryWriter
    from torchvision import transforms as T, datasets
    from torchvision.transforms import autoaugment
    from torchvision.transforms.functional import InterpolationMode

    from models.create_models import create_model
    from utils.plots import plot_datasets, plot_txt, plot_lr_scheduler, plot_loss, plot_confusion_matrix
    from utils.confusion import ConfusionMatrix
    from utils.general import create_config, increment_path, load_train_weight, load_config, ExponentialMovingAverage
    from utils.distributed_utils import init_distributed_mode, cleanup, reduce_value, is_main_process
    from utils.transform import choose_collate_fn

    cfg = init_distributed_mode(cfg)
    device = torch.device(cfg['device'])

    if cfg['config_path'] != '':
        cfg = load_config(cfg['config_path'])

    model_name = cfg['model_prefix'] + '_' + cfg['model_suffix']
    log_dir = increment_path(os.path.join(cfg['log_root'], model_name, cfg['exp_name']))

    if is_main_process():
        os.makedirs(log_dir, exist_ok=True)
        plot_datasets(cfg['img_path'], log_dir)
    results_file = os.path.join(log_dir, 'results.txt')
    train_root = os.path.join(cfg['img_path'], "train")
    val_root = os.path.join(cfg['img_path'], "val")

    if is_main_process():
        tb_writer = SummaryWriter(log_dir=log_dir)
        create_config(log_dir, cfg)

    if cfg['use_benchmark']:
        torch.backends.cudnn.benchmark = True

    if is_main_process():
        print('[INFO] Logs will be saved in {}...'.format(log_dir))

        class_index = ' '.join(['%7s' % (str(i)) for i in np.arange(cfg['num_classes'])])
        with open(results_file, 'w') as f:
            f.write('epoch ' + 'accuracy ' + 'precision ' + 'recall ' + 'F1-score ' + \
                    class_index + ' ' + class_index + ' ' + class_index)

    IpMode = InterpolationMode.BILINEAR
    train_compose = [T.RandomResizedCrop(cfg['img_size'], interpolation=IpMode),
                     T.RandomHorizontalFlip(p=cfg['horizontal_flip'])]
    if cfg['augment'] == 'ra':
        train_compose.append(autoaugment.RandAugment(interpolation=IpMode))
    elif cfg['augment'] == 'simple':
        pass
    elif cfg['augment'] == 'tawide':
        train_compose.append(autoaugment.TrivialAugmentWide(interpolation=IpMode))
    else:
        train_compose.append(
            autoaugment.AutoAugment(policy=autoaugment.AutoAugmentPolicy(cfg['augment']), interpolation=IpMode))

    train_compose.extend([T.PILToTensor(), T.ConvertImageDtype(torch.float),
                          T.Normalize(cfg['mean'], cfg['std']), T.RandomErasing(p=cfg['random_erase_prob'])])
    data_transform = {"train": T.Compose(train_compose),
                      "val": T.Compose([T.Resize(cfg['val_resize'], IpMode),
                                        T.CenterCrop(cfg['img_size']),
                                        T.PILToTensor(),
                                        T.ConvertImageDtype(torch.float),
                                        T.Normalize(cfg['mean'], cfg['std'])])}

    train_dataset = datasets.ImageFolder(root=train_root, transform=data_transform["train"])
    train_num = len(train_dataset)
    train_sampler = DistributedSampler(train_dataset)
    train_batch_sampler = BatchSampler(train_sampler, cfg['batch_size'], drop_last=True)
    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler,
                              persistent_workers=cfg['persistent_workers'], pin_memory=cfg['pin_memory'],
                              num_workers=cfg['num_workers'],
                              collate_fn=choose_collate_fn(cfg['num_classes'], cfg['mixup_prob'], cfg['cutmix_prob']).collate_fn)

    validate_dataset = datasets.ImageFolder(root=val_root, transform=data_transform["val"])
    val_num = len(validate_dataset)
    val_sampler = DistributedSampler(validate_dataset)
    validate_loader = DataLoader(validate_dataset, batch_size=cfg['batch_size'], sampler=val_sampler,
                                 persistent_workers=cfg['persistent_workers'], pin_memory=cfg['pin_memory'],
                                 num_workers=cfg['num_workers'])

    if is_main_process():
        print('[INFO] Load Image From {}...'.format(cfg['img_path']))

        # write dict into json file
        labels_list = train_dataset.class_to_idx
        cla_dict = dict((val, key) for key, val in labels_list.items())
        json_str = json.dumps(cla_dict, indent=4)
        with open(os.path.join(log_dir, 'class_indices.json'), 'w') as json_file:
            json_file.write(json_str)
        labels_name = list(cla_dict.values())
        if len(labels_name) != cfg['num_classes']:
            raise ValueError(
                '[INFO] Find {} classes but configs show {} num_classes'.format(len(labels_name), cfg['num_classes']))

        print('[INFO] {} to train, {} to val, total {} classes...'.format(train_num, val_num, cfg['num_classes']))
    net = create_model(model_name=model_name, num_classes=cfg['num_classes'])

    net = net.to(device)
    if cfg['distributed'] and cfg['sync_bn']:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    loss_function = nn.CrossEntropyLoss(label_smoothing=cfg['smoothing'])

    if cfg['optimizer_type'] == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=cfg['init_lr'], momentum=cfg['momentum'], nesterov=cfg['nesterov'],
                              weight_decay=cfg['weight_decay'])
    elif cfg['optimizer_type'] == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=cfg['init_lr'], weight_decay=cfg['weight_decay'],
                               betas=cfg['betas'], eps=cfg['eps'])
    elif cfg['optimizer_type'] == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr=cfg['init_lr'], weight_decay=cfg['weight_decay'],
                                betas=cfg['betas'], eps=cfg['eps'])
    elif cfg['optimizer_type'] == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), lr=cfg['init_lr'], momentum=cfg['momentum'],
                                  weight_decay=cfg['weight_decay'], eps=cfg['eps'])
    else:
        raise ValueError(
            'Unsupported optimizer_type - `{}`. Only sgd, adam, adamw, rmsprop'.format(cfg['optimizer_type']))

    if cfg['scheduler_type'] == "step_lr":
        gamma = 0.1
        main_lr_scheduler = MultiStepLR(optimizer, milestones=cfg['steps'], gamma=gamma)
    elif cfg['scheduler_type'] == "cosine_lr":
        gamma = 0
        main_lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs'] - cfg['warmup_epochs'], eta_min=1e-7)
    elif cfg['scheduler_type'] == "exponential_lr":
        gamma = 0.9
        main_lr_scheduler = ExponentialLR(optimizer, gamma=gamma)
    else:
        raise ValueError(
            'Unsupported scheduler_type - {}. Only step_lr, cosine_lr, exponential_lr are supported.'.format(
                cfg['scheduler_type']))

    if cfg['warmup_epochs'] > 0:
        if cfg['warmup_type'] == 'linear':
            warmup_decay = 0.05
            warmup_lr_scheduler = LinearLR(optimizer, start_factor=warmup_decay, total_iters=cfg['warmup_epochs'])
        elif cfg['warmup_type'] == "constant":
            warmup_decay = 1.0 / 3
            warmup_lr_scheduler = ConstantLR(optimizer, factor=warmup_decay, total_iters=cfg['warmup_epochs'])
        else:
            raise ValueError(
                'Unsupported warmup_type - {}. Only linear, constant are supported.'.format(cfg['warmup_type']))
        scheduler = SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler],
                                 milestones=[cfg['warmup_epochs']])
    else:
        warmup_decay = None
        scheduler = main_lr_scheduler

    scaler = torch.cuda.amp.GradScaler() if cfg['use_apex'] else None

    net_without_ddp = net
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[cfg['gpu']])
    net_without_ddp = net.module

    net_ema = None
    if cfg['use_ema']:
        ema_decay = 0.99998
        ema_steps = 32
        # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = cfg['world_size'] * cfg['batch_size'] * ema_steps / cfg['epochs']
        alpha = 1.0 - ema_decay
        alpha = min(1.0, alpha * adjust)
        net_ema = ExponentialMovingAverage(net_without_ddp, device=device, decay=1.0 - alpha)

    if cfg['load_from'] != "":
        print('[INFO] Load Weight From {}...'.format(cfg['load_from']))
        if os.path.exists(cfg['load_from']):
            load_train_weight(net_without_ddp, cfg['load_from'], optimizer, scheduler, scaler, net_ema, cfg)
        else:
            raise FileNotFoundError("[INFO] Not found weights file: {}...".format(cfg['load_from']))

    start_epoch = scheduler.last_epoch
    plot_lr_scheduler(cfg['warmup_type'], cfg['optimizer_type'], cfg['scheduler_type'], net, cfg['init_lr'],
                      start_epoch, cfg['steps'], gamma, warmup_decay, cfg['warmup_epochs'], cfg['epochs'], log_dir)

    if is_main_process():
        best_acc = 0.0
        train_loss_list = []
        val_loss_list = []




    if cfg['only_val']:
        # validate
        net.eval()
        confusion = ConfusionMatrix(num_classes=cfg['num_classes'], device=device)
        with torch.no_grad():
            val_per_epoch_loss = torch.zeros(1).to(device)
            if is_main_process():
                validate_loader = tqdm(validate_loader)
            for step, (val_images, val_labels) in enumerate(validate_loader):
                val_images, val_labels = val_images.to(device, non_blocking=True), val_labels.to(device,
                                                                                                 non_blocking=True)
                if net_ema:
                    outputs = net_ema(val_images)
                else:
                    outputs = net(val_images)
                val_loss = loss_function(outputs, val_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                confusion.update(val_labels, predict_y)
                confusion.acc_p_r_f1()

                confusion.mean_val_accuracy = reduce_value(confusion.mean_val_accuracy, average=True)
                confusion.mean_precision = reduce_value(confusion.mean_precision, average=True)
                confusion.mean_recall = reduce_value(confusion.mean_recall, average=True)
                confusion.mean_F1 = reduce_value(confusion.mean_F1, average=True)

                val_loss = reduce_value(val_loss, average=True)
                val_per_epoch_loss = (val_per_epoch_loss * step + val_loss.detach()) / (step + 1)

                if is_main_process():
                    validate_loader.desc = '{}val: val_loss:{:.3f} Acc: {:.3f} ' \
                                           'P: {:.3f} R: {:.3f} F1: {:.3f}'.format('ema' if cfg['use_ema'] else '',
                                                                                   val_loss.detach(),
                                                                                   confusion.mean_val_accuracy,
                                                                                   confusion.mean_precision,
                                                                                   confusion.mean_recall,
                                                                                   confusion.mean_F1)
            if is_main_process():
                val_loss_list.append(val_per_epoch_loss.item())
            if device != torch.device("cpu"):
                torch.cuda.synchronize(device)
        sys.exit()

    if is_main_process():
        print('[INFO] Start Training...')

    start_time = time()
    for epoch in range(scheduler.last_epoch, cfg['epochs']):
        train_sampler.set_epoch(epoch)
        net.train()
        train_per_epoch_loss = torch.zeros(1).to(device)
        if is_main_process():
            train_loader = tqdm(train_loader)
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = net(images)
                loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if cfg['clip_grad']:
                    # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                    scaler.unscale_(optimizer)
                    clip_grad_norm = 1
                    nn.utils.clip_grad_norm_(net.parameters(), clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg['clip_grad']:
                    clip_grad_norm = 1
                    nn.utils.clip_grad_norm_(net.parameters(), clip_grad_norm)
                optimizer.step()

            if net_ema and step % ema_steps == 0:
                net_ema.update_parameters(net)
                if epoch < cfg['warmup_epochs']:
                    # Reset ema buffer to keep copying weights during warmup period
                    net_ema.n_averaged.fill_(0)

            # print statistics
            loss = reduce_value(loss, average=True)
            train_per_epoch_loss = (train_per_epoch_loss * step + loss.detach()) / (step + 1)

            if is_main_process():
                train_loader.desc = "train epoch[{}/{}] train_loss:{:.3f} lr:{:.6f}".format(epoch + 1, cfg['epochs'],
                                                                                            loss.detach(),
                                                                                            optimizer.param_groups[0][
                                                                                                "lr"])

        if is_main_process():
            train_loss_list.append(train_per_epoch_loss.item())
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        # validate
        net.eval()
        confusion = ConfusionMatrix(num_classes=cfg['num_classes'], device=device)
        with torch.no_grad():
            val_per_epoch_loss = torch.zeros(1).to(device)
            if is_main_process():
                validate_loader = tqdm(validate_loader)
            for step, (val_images, val_labels) in enumerate(validate_loader):
                val_images, val_labels = val_images.to(device, non_blocking=True), val_labels.to(device,
                                                                                                 non_blocking=True)
                if net_ema:
                    outputs = net_ema(val_images)
                else:
                    outputs = net(val_images)
                val_loss = loss_function(outputs, val_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                confusion.update(val_labels, predict_y)
                confusion.acc_p_r_f1()

                confusion.mean_val_accuracy = reduce_value(confusion.mean_val_accuracy, average=True)
                confusion.mean_precision = reduce_value(confusion.mean_precision, average=True)
                confusion.mean_recall = reduce_value(confusion.mean_recall, average=True)
                confusion.mean_F1 = reduce_value(confusion.mean_F1, average=True)

                val_loss = reduce_value(val_loss, average=True)
                val_per_epoch_loss = (val_per_epoch_loss * step + val_loss.detach()) / (step + 1)

                if is_main_process():
                    validate_loader.desc = '{}val epoch[{}/{}] val_loss:{:.3f} Acc: {:.3f} ' \
                                           'P: {:.3f} R: {:.3f} F1: {:.3f}'.format('ema' if cfg['use_ema'] else '',
                                                                                   epoch + 1,
                                                                                   cfg['epochs'], val_loss.detach(),
                                                                                   confusion.mean_val_accuracy,
                                                                                   confusion.mean_precision,
                                                                                   confusion.mean_recall,
                                                                                   confusion.mean_F1)
            if is_main_process():
                val_loss_list.append(val_per_epoch_loss.item())
            if device != torch.device("cpu"):
                torch.cuda.synchronize(device)

        scheduler.step()

        if is_main_process():
            tb_writer.add_scalar('train_loss', train_per_epoch_loss, epoch + 1)
            tb_writer.add_scalar('val_loss', val_per_epoch_loss, epoch + 1)
            tb_writer.add_scalar('val_accuracy', confusion.mean_val_accuracy, epoch + 1)
            tb_writer.add_scalar('val_precision', confusion.mean_precision, epoch + 1)
            tb_writer.add_scalar('val_recall', confusion.mean_recall, epoch + 1)
            tb_writer.add_scalar('val_F1', confusion.mean_F1, epoch + 1)

            if confusion.mean_val_accuracy > best_acc:
                best_acc = confusion.mean_val_accuracy
                checkpoint = {'epoch': epoch, 'net': net.module.state_dict(), 'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}
                if scaler:
                    checkpoint["scaler"] = scaler.state_dict()
                if net_ema:
                    checkpoint["net_ema"] = net_ema.state_dict()
                torch.save(checkpoint, log_dir + '/best.pth')
            if epoch + 1 != cfg['epochs']:
                confusion.save(results_file, epoch + 1)
            else:
                confusion_matrix = confusion.confusionmat.cpu().numpy()
                confusion.save(results_file, epoch + 1)

    if is_main_process():
        checkpoint = {'epoch': cfg['epochs'], 'net': net.module.state_dict(), 'optimizer': optimizer.state_dict(),
                      'scheduler': scheduler.state_dict()}
        if scaler:
            checkpoint["scaler"] = scaler.state_dict()
        if net_ema:
            checkpoint["net_ema"] = net_ema.state_dict()

        torch.save(checkpoint, log_dir + '/last.pth')
        plot_txt(log_dir, cfg['num_classes'], labels_name)
        plot_loss(log_dir, train_loss_list, val_loss_list)
        plot_confusion_matrix(confusion_matrix, log_dir)
        print('[INFO] Results will be saved in {}...'.format(log_dir))
        print('[INFO] Finish Training...Cost time: %ss' % (time() - start_time))
        print(cfg)
    cleanup()


if __name__ == '__main__':
    from train_config import configurations
    import os
    import sys
    sys.setrecursionlimit(1000000)

    cfg = configurations['cfg']
    if cfg['device'] == 'cuda':
        assert len(cfg['gpu_ids'].split(',')) == cfg['world_size']
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg['gpu_ids']

    main(cfg)
