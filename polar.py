from argparse import ArgumentParser
from collections import namedtuple
from itertools import zip_longest

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import utils
import analysis

# Named tuples
Batch = namedtuple("Batch", field_names=["image", "radius", "phi"])


def scaled_softsign(tensor):
    """
    Computes a scaled version of the softsign function.
    Range of values: [0, 1]
    :param tensor: A PyTorch Tensor
    :return: A new PyTorch Tensor of the same shape and dtype
    """
    return .5 + torch.div(tensor, 2. * (1. + torch.abs(tensor)))


def softsign(tensor):
    """
    Computes the softsign function
    Range of values: [-1, +1]
    :param tensor: A PyTorch Tensor
    :return: A new PyTorch Tensor of the same shape and dtype
    """
    return torch.div(tensor, 1. + torch.abs(tensor))


def great_circle_distance(alpha, beta):
    """
    Computes the unsigned distance between two angles divided by π
    Range of values: [0, 1]
    :param alpha: A PyTorch Tensor that specifies the first angle
    :param beta: A PyTorch Tensor that specifies the second angle
    :return: A new PyTorch Tensor of the same shape and dtype
    """
    abs_diff = torch.abs(beta - alpha)
    return torch.where(abs_diff <= np.pi,
                       abs_diff,
                       2. * np.pi - abs_diff)/np.pi


class Network(nn.Module):
    def __init__(self, network_config, in_height, in_width):
        """
        :param network_config:
            A dictionary of the following structure:
                - cnn (itself a dictionary of the following keys):
                    - activations:
                    - kernel_sizes:
                    - strides:
                    - num_channels:
                - fc (itself a dictionary of the following keys):
                    - activations:
                    - num_units:
        :param in_height: The vertical dimension of the input images
        :param in_width: The horizontal dimension of the input images
        """
        super().__init__()

        # Split configurations
        cnn_config = network_config["cnn"]
        fc_config = network_config["fc"]

        # Copy
        self.cnn_activations = list(map(utils.resolve_activations, cnn_config["activations"]))
        self.cnn_kernel_sizes = cnn_config["kernel_sizes"]
        self.cnn_strides = cnn_config["strides"]
        self.cnn_in_channels = [3] + cnn_config["num_channels"][:-1]
        self.cnn_out_channels = cnn_config["num_channels"]

        # Convolutional network
        self.cnn, h, w = utils.build_convolutional_network(in_channels=self.cnn_in_channels,
                                                           out_channels=self.cnn_out_channels,
                                                           kernel_sizes=self.cnn_kernel_sizes,
                                                           strides=self.cnn_strides,
                                                           activations=cnn_config["activations"],
                                                           in_height=in_height, in_width=in_width)

        # Instance normalization
        self.instance_norms = [nn.InstanceNorm2d(num_features=cnn_out_channel,
                                                 track_running_stats=True) for cnn_out_channel in
                               self.cnn_out_channels]

        # Dense network
        in_features = self.cnn_out_channels[-1] * h * w
        self.fc_num_layers = len(fc_config["activations"]) + 1
        self.fc_activations = list(map(utils.resolve_activations, fc_config["activations"]))
        self.fc_num_units = [in_features] + fc_config["num_units"] + [2]
        self.fc = utils.build_dense_network(num_units=self.fc_num_units, activations=fc_config["activations"])

        # Batch normalization
        self.batch_norms = [nn.BatchNorm1d(num_features=num_units,
                                           affine=False) for num_units in self.fc_num_units[1:]]

        # Lambda (scale) of activation function
        self.activation_lambda = network_config["activation_lambda"]
        self.activation_radius = nn.Sigmoid()
        self.activation_angle = nn.Tanh()

        # Make PyTorch aware of sub-networks
        self.cnn = nn.ModuleList(self.cnn)
        self.instance_norms = nn.ModuleList(self.instance_norms)
        self.fc = nn.ModuleList(self.fc)
        self.batch_norms = nn.ModuleList(self.batch_norms)

    def forward(self, image):
        """
        Computes the forward pass of the network.
        :param image: A PyTorch Tensor of shape [N, C, H, W]
        :return: A PyTorch Tensor of shape [N, 2] where the first column contains predicted radii and the second
        contains predicted angles
        """

        # Pass through convolutional layers
        x = image
        for cnn_layer, activation, instance_norm in zip(self.cnn,
                                                        self.cnn_activations,
                                                        self.instance_norms):
            x = cnn_layer(x)
            x = activation(x)
            x = instance_norm(x)

        # Flatten last two dimensions
        x = x.view(x.shape[0], -1)  # [B, C * H * W]

        # Pass through fully-connected layers
        for fc_layer, activation, batch_norm in zip_longest(self.fc, self.fc_activations, self.batch_norms):
            x = fc_layer(x)

            if activation is not None:
                x = activation(x)
            if batch_norm is not None:
                x = batch_norm(x)

        # Final activations
        radius_pred = self.activation_radius(self.activation_lambda * x[:, 0])
        angle_pred = np.pi * self.activation_angle(self.activation_lambda/2. * x[:, 1])

        return radius_pred, angle_pred


def gen(batch_size, width, height, circle_radius, device):
    """
    A generator that yields batches
    :param batch_size: An arbitrary batch size > 1
    :param width: The horizontal dimension of the input images
    :param height: The vertical dimension of the input images
    :param circle_radius: The size of the two points to be drawn
    :param device: A PyTorch device (e.g. "cuda:0" or "cpu")
    :yields Named "Batch" tuples that contain the image as well as the ground truth
    """

    # Spatial
    center_x = width // 2
    center_y = height // 2
    max_radial_dist = np.sqrt(center_x**2 + center_y**2)

    while True:

        # Random cartesian coordinates
        x = np.random.uniform(low=-center_x + circle_radius, high=center_x - circle_radius, size=batch_size)
        y = np.random.uniform(low=-center_y + circle_radius, high=center_y - circle_radius, size=batch_size)

        # Draw points
        x = x.astype(np.int32)
        y = y.astype(np.int32)
        img_batch = []
        for i in range(batch_size):
            img = np.ones(shape=[height, width, 3], dtype=np.float32)    # [H, W, C]
            img = cv2.circle(img=img.copy(),
                             center=(center_x, center_y),
                             radius=circle_radius,
                             color=[0, 0, 0], thickness=cv2.FILLED)
            img = cv2.circle(img=img,
                             center=(center_x + x[i], center_y - y[i]),
                             radius=circle_radius,
                             color=[0, 1, 0], thickness=cv2.FILLED)
            img_batch.append(img)

        # Stack
        img_batch = np.stack(img_batch, axis=0)

        # Revert dimensions
        img_batch = np.transpose(img_batch, axes=[0, 3, 1, 2])

        # Polar coordinates (radii are normalized w.r.t maximum radius)
        cartesian = np.stack((x, y), axis=1)
        radius = np.linalg.norm(cartesian, axis=1, ord=2) / max_radial_dist
        phi = np.arctan2(y, x)

        # Copy onto device
        img_batch = torch.tensor(img_batch, dtype=torch.float32, device=device)
        radius = torch.tensor(radius, dtype=torch.float32, device=device)
        phi = torch.tensor(phi, dtype=torch.float32, device=device)

        yield Batch(img_batch, radius, phi)


def main():
    # Instantiate parser
    parser = ArgumentParser()

    parser.add_argument("--network_config", required=True)

    # Parse
    args = parser.parse_args()

    # Parse config
    config = utils.parse_config(config_fname=args.network_config)

    # Temporary files
    tmp_dir = utils.create_temp_dir(config_fname=args.network_config)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TensorBoard
    summary_writer = SummaryWriter(tmp_dir)

    # Test batch (for visualization purposes)
    test_batch = analysis.get_test_batch(batch_size=config["batch_size"],
                                         width=config["width"],
                                         height=config["height"],
                                         circle_radius=config["circle_radius"],
                                         device=device)

    # Build network
    net = Network(config["network"],
                  in_height=config["height"],
                  in_width=config["width"]).to(device)

    # Optimizer
    optimizer = optim.SGD(net.parameters(),
                          lr=config["learning_rate"],
                          momentum=config["momentum"],
                          nesterov=config["nesterov"])

    # Learning rate schedule
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=config["steplr_step_size"],
                                          gamma=config["steplr_gamma"])

    # Loss functions
    criterion_radius = nn.L1Loss()

    def _criterion_phi(alpha, beta):
        """
        PyTorch training criterion for angle loss (mean of great circle distances)
        Range of values: [0, 1]
        :param alpha: A PyTorch Tensor specifying the first angles
        :param beta: A PyTorch Tensor specifying the second angles
        :return: A new PyTorch Tensor of same shape and dtype
        """
        return great_circle_distance(alpha, beta).mean()

    for global_step, batch in enumerate(gen(batch_size=config["batch_size"],
                                            height=config["height"],
                                            width=config["width"],
                                            circle_radius=config["circle_radius"],
                                            device=device)):

        optimizer.zero_grad()
        radius_pred, phi_pred = net(batch.image)
        loss_radius = criterion_radius(radius_pred, batch.radius)
        loss_phi = _criterion_phi(phi_pred, batch.phi)
        loss = (loss_radius + loss_phi)/2.
        loss.backward()
        optimizer.step()

        # Adjust learning rate
        scheduler.step()

        # TensorBoard
        with torch.set_grad_enabled(mode=False):

            # Copy onto CPU
            radius_pred = radius_pred.cpu().numpy()
            phi_pred = phi_pred.cpu().numpy()

            summary_writer.add_scalar(tag="l1_radius", scalar_value=loss_radius, global_step=global_step)
            summary_writer.add_scalar(tag="l1_phi", scalar_value=loss_phi, global_step=global_step)
            summary_writer.add_scalar(tag="l1", scalar_value=loss, global_step=global_step)

            summary_writer.add_histogram(tag="histogram_radius_pred", values=radius_pred, global_step=global_step)
            summary_writer.add_histogram(tag="histogram_phi_pred", values=phi_pred, global_step=global_step)

            # Draw
            if config["save_images"]:

                # Eval mode
                net = net.eval()

                # Copy onto CPU
                radius_test_pred, phi_test_pred = net(test_batch.image)
                radius_test_pred = radius_test_pred.cpu().numpy()
                phi_test_pred = phi_test_pred.cpu().numpy()

                analysis.draw_test_batch(image_batch=test_batch.image.cpu().numpy(),
                                         radius_batch=radius_test_pred,
                                         phi_batch=phi_test_pred,
                                         tmp_dir=tmp_dir,
                                         global_step=global_step,
                                         height=config["height"],
                                         width=config["width"],
                                         circle_radius=config["circle_radius"])

                # Training mode
                net = net.train()


if __name__ == "__main__":
    main()
