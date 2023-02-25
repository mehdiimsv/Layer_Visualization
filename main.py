from PIL import Image
import argparse
import yaml
from utils import *


def main(config, image):

    data_config, model_config, device = get_config(config)
    transform = make_transform(data_config)
    image = preprocess(image, transform, device)
    model = get_model(model_config, device)
    conv_layers = get_conv_layers(model)
    outputs, names = get_output(conv_layers, image)
    processed = get_mean_value_of_each_output(outputs)
    plot_and_save(processed, names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get config file')
    parser.add_argument('--config_path', required=False, default='./config.yaml',
                        help='Config file path')
    parser.add_argument('--input_img', required=False, default='./dog.jpg',
                        help='Config file path')

    args = parser.parse_args()

    with open(args.config_path) as file:
        config_file = yaml.full_load(file)

    image_file = Image.open(args.input_img)
    main(config_file, image_file)
