from torchvision import models, transforms
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys


def get_config(config):
    data_config = config['data']['augmentation']
    model_config = config['model']
    device = config['device']
    return data_config, model_config, device


def make_transform(tools):
    resize_param = tools['resize']
    mean_param, std_param = tools['normalize']['mean'], tools['normalize']['std']
    transform = transforms.Compose([
        transforms.Resize((resize_param, resize_param)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_param, std=std_param)
    ])
    return transform


def preprocess(image, transform, device):
    print(f"Image shape before: {np.shape(image)}")
    image = transform(image)
    image = image.unsqueeze(0)
    print(f"Image shape after: {image.shape}")
    image = image.to(device)
    return image


def get_model(tools, device):
    model_name = tools['name']

    if model_name.lower() == 'resnet18':
        net = models.resnet18(pretrained=True)
    elif model_name.lower() == 'resnet34':
        net = models.resnet34(pretrained=True)
    elif model_name.lower() == 'resnet50':
        net = models.resnet50(pretrained=True)
    elif model_name.lower() == 'resnet101':
        net = models.resnet101(pretrained=True)
    elif model_name.lower() == 'resnet152':
        net = models.resnet152(pretrained=True)
    else:
        print(f'Your input model name {model_name} is not resnet version')
        sys.exit()
    model = net.to(device)
    return model


def get_conv_layers(model):
    model_weights = []
    conv_layers = []
    model_children = list(model.children())

    counter = 0
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)

    print(f"Total convolution layers: {counter}")
    print(conv_layers)
    return conv_layers


def get_output(conv_layers, image):
    outputs = []
    names = []
    for layer in conv_layers:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))

    print(len(outputs))
    for feature_map in outputs:
        print(feature_map.shape)

    return outputs, names


def get_mean_value_of_each_output(outputs):
    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())
    for fm in processed:
        print(fm.shape)

    return processed


def plot_and_save(processed, names):
    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed)):
        a = fig.add_subplot(12, int(len(names) / 5) + 1, i + 1)
        imgplot = plt.imshow(processed[i])
        a.axis("off")
        a.set_title(names[i].split('(')[0], fontsize=10)
    plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')
