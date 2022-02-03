if __name__ == '__main__':
    import os
    import json
    import torch
    from PIL import Image
    from torchvision import transforms
    import matplotlib.pyplot as plt
    from predict_config import configurations
    from models.create_models import create_model
    from utils.general import load_predict_weight

    cfg = configurations['cfg']
    load_from = cfg['load_from']
    predict_img_path = cfg['predict_img_path']
    mean = cfg['mean']
    std = cfg['std']
    img_size = cfg['img_size']
    num_classes = cfg['num_classes']
    device = cfg['device']
    model_prefix = cfg['model_prefix']
    model_suffix = cfg['model_suffix']
    class_indices_path = cfg['class_indices_path']
    model_name = model_prefix + '_' + model_suffix

    data_transform = transforms.Compose([transforms.Resize((int(img_size[0]*1.2), int(img_size[0]*1.2))),
                                         transforms.CenterCrop(img_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean, std)])

    # load image
    assert os.path.exists(predict_img_path), "File: '{}' dose not exist.".format(predict_img_path)
    img = Image.open(predict_img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = class_indices_path
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    net = create_model(model_name=model_name, num_classes=num_classes).to(device)
    # load model weights

    if load_from != "":
        print('[INFO] Load Weight From {}...'.format(load_from))
        if os.path.exists(load_from):
            load_predict_weight(net, load_from)
        else:
            raise FileNotFoundError("[INFO] Not found weights file: {}...".format(load_from))
        print('[INFO] Successfully Load Weight From {}...'.format(load_from))

    net.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(net(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()


