import pathlib
import numpy as np
import cv2

from polar import gen


def get_test_batch(batch_size, width, height, circle_radius, device):
    return next(gen(batch_size=batch_size,
                    width=width,
                    height=height,
                    circle_radius=circle_radius,
                    device=device))


def draw_test_batch(image_batch, radius_batch, phi_batch, tmp_dir, global_step, width, height, circle_radius):
    """
    Draws and saves a test batch and its predictions
    :param image_batch: A PyTorch Tensor of shape [N, C, H, W]
    :param radius_batch: A PyTorch Tensor of shape [N]
    :param phi_batch: A PyTorch Tensor of shape [N]
    :param tmp_dir: The temporary directory used for saving the images
    :param global_step: The current global step
    :param width: Width of the images
    :param height: Height of the images
    :param circle_radius: Radius of the circle to be drawn
    """

    # Re-order image dimensions
    image_batch = np.transpose(image_batch, axes=[0, 2, 3, 1])

    # Spatial
    center_x = width // 2
    center_y = height // 2
    max_radial_dist = np.sqrt(center_x**2 + center_y**2)
    padding_width = int(max_radial_dist) - center_x
    padding_height = int(max_radial_dist) - center_y

    # Output
    img_path = pathlib.Path(tmp_dir).joinpath("images")

    for i, (image, radius, phi) in enumerate(zip(image_batch, radius_batch, phi_batch)):

        # Cartesian coordinates
        x = (max_radial_dist * radius * np.cos(phi)).astype(np.int32)
        y = (max_radial_dist * radius * np.sin(phi)).astype(np.int32)

        # Pad images to ``max_radial_dist`` from both sides
        image_padded = np.pad(image.copy(),
                              pad_width=[(padding_height, padding_height),
                                         (padding_width, padding_width),
                                         (0, 0)],
                              mode="constant",
                              constant_values=1)

        # Draw prediction
        overlay = cv2.circle(img=image_padded,
                             center=(padding_width + center_x + x, padding_height + center_y - y),
                             radius=circle_radius,
                             color=(0, 0, 0))

        # Write out
        instance_path = img_path.joinpath(str(i))
        instance_path.mkdir(exist_ok=True)
        cv2.imwrite(filename=str(instance_path.joinpath("{}.png".format(global_step))), img=(2**8 - 1) * overlay)
