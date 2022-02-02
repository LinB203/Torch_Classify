if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    import json
    from torch.utils.data import DataLoader
    from models.create_models import create_model
    import torch
    from torchvision import transforms, datasets
    import os
    from tqdm import tqdm
    from utils.confusion import ConfusionMatrix
    from torch.utils.tensorboard import SummaryWriter
    from utils.scheduler import get_lr, create_scheduler
    from utils.optimizer import create_optimizer
    from utils.plots import plot_datasets, plot_txt, plot_lr_scheduler, plot_loss, plot_confusion_matrix
    from utils.loss import create_loss
    from utils.general import create_config, increment_path, load_train_weight, load_config
    from train_config import configurations
    import numpy as np

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
    print('[INFO] Using Model:{} Epoch:{} BatchSize:{} LossType:{} '
          'OptimizerType:{} SchedulerType:{}...'.format(model_name, epochs, batch_size,
                                                        loss_type, optimizer_type, scheduler_type))
    print('[INFO] Logs will be saved in {}...'.format(log_dir))

    class_index = ' '.join(['%7s'%(str(i)) for i in np.arange(num_classes)])
    with open(results_file, 'w') as f:
        f.write('epoch ' + 'accuracy ' + 'precision ' + 'recall ' + 'F1-score ' + \
                class_index + ' ' + class_index + ' ' + class_index)

    print("[INFO] Using {} device...".format(device))

    data_transform = {"train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean, std)]),
                      "val": transforms.Compose([transforms.Resize(img_size),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean, std)])}
    if num_workers =='auto':
        num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('[INFO] Using {} dataloader workers every process'.format(num_workers))

    train_dataset = datasets.ImageFolder(root=train_root, transform=data_transform["train"])
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               pin_memory=True, num_workers=num_workers)

    validate_dataset = datasets.ImageFolder(root=val_root, transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False,
                                                  pin_memory=True, num_workers=num_workers)
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

    loss_function = create_loss(loss_type, alpha=alpha,
                                gamma=gamma, num_classes=num_classes)

    optimizer = create_optimizer(optimizer_type, net, init_lr)
    scheduler = create_scheduler(scheduler_type, optimizer, epochs, steps, warmup_epochs)
    # print(scheduler.last_epoch)
    if load_from != "":
        print('[INFO] Load Weight From {}...'.format(load_from))
        if os.path.exists(load_from):
            net, optimizer, start_epoch = load_train_weight(net, load_from, optimizer, scheduler.last_epoch)
            scheduler.last_epoch = start_epoch
        else:
            raise FileNotFoundError("[INFO] Not found weights file: {}...".format(load_from))
        print('[INFO] Successfully Load Weight From {}...'.format(load_from))
    else:
        start_epoch = 0
        scheduler.last_epoch = start_epoch

    plot_lr_scheduler(optimizer_type, scheduler_type, net, init_lr, start_epoch, steps, warmup_epochs, epochs, log_dir)

    if use_apex:
        from apex import amp
        net, optimizer = amp.initialize(net, optimizer, opt_level='O1')
        print('[INFO] Using Mixed-precision to train...')
    best_acc = 0.0
    train_steps = len(train_loader)
    val_steps = len(validate_loader)
    train_loss_list = []
    val_loss_list = []
    print('[INFO] Start Training...')

    for epoch in range(start_epoch, epochs):
        net.train()
        train_per_epoch_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, (images, labels) in enumerate(train_bar):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            if use_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            # print statistics
            train_per_epoch_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] train_loss:{:.3f} lr:{:.6f}".format(epoch + 1, epochs, loss, get_lr(optimizer))
            optimizer.step()

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
            torch.save(checkpoint, log_dir + '/best.pth')
        if epoch+1 != epochs:
            confusion.save(results_file, epoch + 1)
        else:
            confusion_matrix = confusion.confusionmat.int().numpy()
            confusion.save(results_file, epoch + 1)

    checkpoint = {'epoch': epochs, 'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
    torch.save(checkpoint, log_dir + '/last.pth')
    plot_txt(log_dir, num_classes, labels_name)
    plot_loss(log_dir, train_loss_list, val_loss_list)
    plot_confusion_matrix(confusion_matrix, log_dir)
    print('[INFO] Results will be saved in {}...'.format(log_dir))
    print('[INFO] Finished...')
