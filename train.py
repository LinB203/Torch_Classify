if __name__ == '__main__':
    import os
    import json
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    from tqdm import tqdm
    from time import time

    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torch.utils.data.dataloader import default_collate
    from torch.utils.tensorboard import SummaryWriter

    from torchvision import transforms, datasets
    from torchvision.transforms import autoaugment, transforms
    from torchvision.transforms.functional import InterpolationMode

    from models.create_models import create_model
    from utils.scheduler import get_lr, create_scheduler
    from utils.optimizer import create_optimizer
    from utils.plots import plot_datasets, plot_txt, plot_lr_scheduler, plot_loss, plot_confusion_matrix
    from utils.loss import create_loss
    from utils.confusion import ConfusionMatrix
    from utils.general import create_config, increment_path, load_train_weight, load_config, ExponentialMovingAverage
    from utils.transform import RandomMixup, RandomCutmix
    from train_config import configurations

    cfg = configurations['cfg']
    cfg_path = cfg['config_path']
    if cfg_path != '':
        cfg = load_config(cfg_path)
    load_from = cfg['load_from']
    img_path = cfg['img_path']
    exp_name = cfg['exp_name']
    mean = cfg['mean']
    std = cfg['std']
    img_size = cfg['img_size']
    num_classes = cfg['num_classes']
    batch_size = cfg['batch_size']
    epochs = cfg['epochs']
    num_workers = cfg['num_workers']
    device = torch.device(cfg['device'])
    use_benchmark = cfg['use_benchmark']
    scheduler_type = cfg['scheduler_type']
    model_prefix = cfg['model_prefix']
    model_suffix = cfg['model_suffix']
    init_lr = cfg['init_lr']
    optimizer_type = cfg['optimizer_type']
    log_root = cfg['log_root']
    steps = cfg['steps']
    warmup_epochs = cfg['warmup_epochs']
    loss_type = cfg['loss_type']
    alpha = cfg['alpha']
    gamma = cfg['gamma']
    use_apex = cfg['use_apex']
    mixup = cfg['mixup']
    cutmix = cfg['cutmix']
    augment = cfg['augment']
    use_ema = cfg['use_ema']
    clip_grad = cfg['clip_grad']
    print(cfg)

    model_name = model_prefix + '_' + model_suffix
    log_dir = increment_path(os.path.join(log_root, model_name, exp_name))
    os.makedirs(log_dir, exist_ok=True)
    plot_datasets(img_path, log_dir)
    results_file = os.path.join(log_dir, 'results.txt')
    train_root = os.path.join(img_path, "train")
    val_root = os.path.join(img_path, "val")
    tb_writer = SummaryWriter(log_dir=log_dir)
    create_config(log_dir)

    if use_benchmark:
        torch.backends.cudnn.benchmark = True

    print('[INFO] Logs will be saved in {}...'.format(log_dir))

    class_index = ' '.join(['%7s'%(str(i)) for i in np.arange(num_classes)])
    with open(results_file, 'w') as f:
        f.write('epoch ' + 'accuracy ' + 'precision ' + 'recall ' + 'F1-score ' + \
                class_index + ' ' + class_index + ' ' + class_index)

    print("[INFO] Using {} device...".format(device))
    if num_workers == 'auto':
        num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    train_compose = [transforms.RandomResizedCrop(img_size)]
    if augment == 'ra':
        train_compose.append(autoaugment.RandAugment(interpolation=InterpolationMode.BILINEAR))
    elif augment == 'simple':
        pass
    elif augment == 'tawide':
        train_compose.append(autoaugment.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR))
    else:
        train_compose.append(autoaugment.AutoAugment(policy=autoaugment.AutoAugmentPolicy(augment),
                                                     interpolation=InterpolationMode.BILINEAR))

    train_compose.extend([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float),
                          transforms.Normalize(mean, std), transforms.RandomErasing(p=0.2),
                          transforms.RandomHorizontalFlip(p=0.5)])
    data_transform = {"train": transforms.Compose(train_compose),
                      "val": transforms.Compose([transforms.Resize(img_size),
                                                 transforms.PILToTensor(),
                                                 transforms.ConvertImageDtype(torch.float),
                                                 transforms.Normalize(mean, std)])}

    train_dataset = datasets.ImageFolder(root=train_root, transform=data_transform["train"])
    train_num = len(train_dataset)

    if mixup or cutmix:
        mixup_transforms = []
        if cutmix:
            mixup_transforms.append(RandomCutmix(num_classes, p=1.0, alpha=1.0))
        if mixup:
            mixup_transforms.append(RandomMixup(num_classes, p=1.0, alpha=1.0))
        mixupcutmix = transforms.RandomChoice(mixup_transforms)
        collate_fn = lambda batch: mixupcutmix(*default_collate(batch))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   pin_memory=False, num_workers=num_workers, collate_fn=collate_fn)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   pin_memory=False, num_workers=num_workers)

    validate_dataset = datasets.ImageFolder(root=val_root, transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False,
                                                  pin_memory=False, num_workers=num_workers)
    print('[INFO] Load Image From {}...'.format(img_path))

    # write dict into json file
    labels_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in labels_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open(os.path.join(log_dir, 'class_indices.json'), 'w') as json_file:
        json_file.write(json_str)
    labels_name = list(cla_dict.values())
    if len(labels_name) != num_classes:
        raise ValueError('[INFO] Find {} classes but configs show {} num_classes'.format(len(labels_name), num_classes))

    print('[INFO] {} to train, {} to val, total {} classes...'.format(train_num, val_num, num_classes))
    net = create_model(model_name=model_name, num_classes=num_classes).to(device)

    loss_function = create_loss(loss_type, alpha=alpha, gamma=gamma, num_classes=num_classes)

    optimizer = create_optimizer(optimizer_type, net, init_lr)
    scheduler = create_scheduler(scheduler_type, optimizer, epochs, steps, warmup_epochs)
    scaler = torch.cuda.amp.GradScaler() if use_apex else None

    net_ema = None
    if use_ema:
        ema_decay = 0.99998
        ema_steps = 32
        # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = 1 * batch_size * ema_steps / epochs
        alpha = 1.0 - ema_decay
        alpha = min(1.0, alpha * adjust)
        net_ema = ExponentialMovingAverage(net, device=device, decay=1.0 - alpha)

    if load_from != "":
        print('[INFO] Load Weight From {}...'.format(load_from))
        if os.path.exists(load_from):
            load_train_weight(net, load_from, optimizer, scheduler, scaler, net_ema)
        else:
            raise FileNotFoundError("[INFO] Not found weights file: {}...".format(load_from))

    plot_lr_scheduler(optimizer_type, scheduler_type, net, init_lr, scheduler.last_epoch, steps, warmup_epochs, epochs, log_dir)

    best_acc = 0.0
    train_steps = len(train_loader)
    val_steps = len(validate_loader)
    train_loss_list = []
    val_loss_list = []
    print('[INFO] Start Training...')

    start_time = time()
    for epoch in range(scheduler.last_epoch, epochs):
        net.train()
        train_per_epoch_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, (images, labels) in enumerate(train_bar):
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = net(images)
                loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if clip_grad:
                    # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                    scaler.unscale_(optimizer)
                    clip_grad_norm = 1
                    nn.utils.clip_grad_norm_(net.parameters(), clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if clip_grad:
                    clip_grad_norm = 1
                    nn.utils.clip_grad_norm_(net.parameters(), clip_grad_norm)
                optimizer.step()

            if net_ema and step % ema_steps == 0:
                net_ema.update_parameters(net)
                if epoch < warmup_epochs:
                    # Reset ema buffer to keep copying weights during warmup period
                    net_ema.n_averaged.fill_(0)

            # print statistics
            train_per_epoch_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] train_loss:{:.3f} lr:{:.6f}".format(epoch + 1, epochs, loss, get_lr(optimizer))

        train_per_epoch_loss = train_per_epoch_loss / train_steps
        train_loss_list.append(train_per_epoch_loss)

        # validate
        net.eval()
        val_per_epoch_loss = 0.0
        confusion = ConfusionMatrix(num_classes=num_classes)
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for (val_images, val_labels) in val_bar:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                outputs = net(val_images)
                val_loss = loss_function(outputs, val_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                # print(val_labels, '\n', outputs, '\n', predict_y)
                confusion.update(val_labels, predict_y)
                # confusion.update(val_labels, outputs, predict_y)
                confusion.acc_p_r_f1()
                val_per_epoch_loss += val_loss.item()
                val_bar.desc = 'val epoch[{}/{}] val_loss:{:.3f} Acc: {:.3f} '\
                               'P: {:.3f} R: {:.3f} F1: {:.3f}'.format(epoch + 1, epochs,
                                                                       val_loss,
                                                                       confusion.mean_val_accuracy,
                                                                       confusion.mean_precision,
                                                                       confusion.mean_recall,
                                                                       confusion.mean_F1)
            val_per_epoch_loss = val_per_epoch_loss / val_steps
            val_loss_list.append(val_per_epoch_loss)

        scheduler.step()

        tb_writer.add_scalar('train_loss', train_per_epoch_loss, epoch + 1)
        tb_writer.add_scalar('val_loss', val_per_epoch_loss, epoch + 1)
        tb_writer.add_scalar('val_accuracy', confusion.mean_val_accuracy, epoch + 1)
        tb_writer.add_scalar('val_precision', confusion.mean_precision, epoch + 1)
        tb_writer.add_scalar('val_recall', confusion.mean_recall, epoch + 1)
        tb_writer.add_scalar('val_F1', confusion.mean_F1, epoch + 1)

        if confusion.mean_val_accuracy > best_acc:
            best_acc = confusion.mean_val_accuracy
            checkpoint = {'epoch': epoch, 'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            if net_ema:
                checkpoint["net_ema"] = net_ema.state_dict()
            torch.save(checkpoint, log_dir + '/best.pth')
        if epoch + 1 != epochs:
            confusion.save(results_file, epoch + 1)
        else:
            confusion_matrix = confusion.confusionmat.int().numpy()
            confusion.save(results_file, epoch + 1)

    checkpoint = {'epoch': epochs, 'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
    if scaler:
        checkpoint["scaler"] = scaler.state_dict()
    if net_ema:
        checkpoint["net_ema"] = net_ema.state_dict()
    torch.save(checkpoint, log_dir + '/last.pth')
    plot_txt(log_dir, num_classes, labels_name)
    plot_loss(log_dir, train_loss_list, val_loss_list)
    plot_confusion_matrix(confusion_matrix, log_dir)
    print('[INFO] Results will be saved in {}...'.format(log_dir))
    print('[INFO] Finish Training...Cost time: %ss'%(time()-start_time))
    print(cfg)
