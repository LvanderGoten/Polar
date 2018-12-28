import sys
import os
import yaml
from itertools import zip_longest
from datetime import datetime
import tempfile
import shutil

import torch.nn as nn


def parse_config(config_fname):
    with open(config_fname, "r") as f:
        try:
            config = yaml.load(f)
        except yaml.YAMLError as e:
            print(e)
            sys.exit(1)
    return config


def create_temp_dir(config_fname):
    tmp_dir = os.path.join(tempfile.gettempdir(), "PyTorch_{date:%Y-%m-%d_%H_%M_%S}".format(date=datetime.now()))
    os.makedirs(tmp_dir)
    shutil.copyfile(src=config_fname, dst=os.path.join(tmp_dir, os.path.basename(config_fname)))
    print("Temporary model directory: {}".format(tmp_dir))
    img_dir = os.path.join(tmp_dir, "images")
    os.makedirs(img_dir)
    print("Temporary image directory: {}".format(img_dir))
    return tmp_dir


def build_convolutional_network(in_channels, out_channels,
                                kernel_sizes, strides, activations,
                                in_height, in_width):
    cnn = []
    h, w = in_height, in_width
    for in_channel, out_channel, kernel_size, stride, activation in zip(in_channels, out_channels,
                                                                        kernel_sizes, strides, activations):
        # Spatial dimensions
        h = compute_output_shape(h_in=h,
                                 kernel_size=kernel_size[0],
                                 stride=stride[0])
        w = compute_output_shape(h_in=w,
                                 kernel_size=kernel_size[1],
                                 stride=stride[1])

        # Calculate activation gain (if it exists)
        activation_gain = get_activation_gain(activation)

        # Layer
        cnn_layer = nn.Conv2d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=kernel_size,
                              stride=stride)

        # Xavier initialization
        nn.init.xavier_uniform_(cnn_layer.weight, gain=activation_gain)

        # Append
        cnn.append(cnn_layer)

    return cnn, h, w


def build_dense_network(num_units, activations):
    fc = []
    for in_features, out_features, activation in zip_longest(num_units[:-1], num_units[1:], activations):

        if activation is not None:
            # Calculate activation gain (if it exists)
            activation_gain = get_activation_gain(activation)
        else:
            activation_gain = 1

        # Create layer
        fc_layer = nn.Linear(in_features=in_features,
                             out_features=out_features)

        # Xavier initialization
        nn.init.xavier_uniform_(fc_layer.weight, gain=activation_gain)

        # Append layer
        fc.append(fc_layer)

    return fc


def compute_output_shape(h_in, kernel_size, stride):
    return int((h_in - kernel_size)/stride + 1)


def resolve_activations(s):
    if s.startswith("torch.nn."):
        return getattr(nn, s.replace("torch.nn.", ""))()
    else:
        raise ValueError("Only cnn_activations starting with 'torch.nn' are acceptable!")


def map2functional(activation):
    return activation.split(".")[-1].lower()


def get_activation_gain(activation):
    try:
        activation_gain = nn.init.calculate_gain(nonlinearity=map2functional(activation))
    except ValueError:
        # Gain not available
        activation_gain = 1
    return activation_gain
