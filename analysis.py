import pathlib
import numpy as np
import cv2

from polar import gen


def get_test_batch(batch_size, size, circle_radius, device):
    return next(gen(batch_size=batch_size,
                    size=size,
                    circle_radius=circle_radius,
                    device=device))


def draw_test_batch(image_batch, radius_batch, phi_batch, tmp_dir, global_step, size, circle_radius):
    """
    Draws and saves a test batch and its predictions
    :param image_batch: A PyTorch Tensor of shape [N, C, H, W]
    :param radius_batch: A PyTorch Tensor of shape [N]
    :param phi_batch: A PyTorch Tensor of shape [N]
    :param tmp_dir: The temporary directory used for saving the images
    :param global_step: The current global step
    :param size: Size of the images
    :param circle_radius: Radius of the circle to be drawn
    """

    # Re-order image dimensions
    image_batch = np.transpose(image_batch, axes=[0, 2, 3, 1])

    # Spatial
    center = size // 2

    # Output
    img_path = pathlib.Path(tmp_dir).joinpath("images")

    for i, (image, radius, phi) in enumerate(zip(image_batch, radius_batch, phi_batch)):

        # Cartesian coordinates
        x = ((center - circle_radius) * radius * np.cos(phi)).astype(np.int32)
        y = ((center - circle_radius) * radius * np.sin(phi)).astype(np.int32)

        # Draw prediction
        overlay = cv2.circle(img=image.copy(),
                             center=(center + x, center - y),
                             radius=circle_radius,
                             color=(1, 0, 0),
                             thickness=cv2.FILLED)

        # Write out
        instance_path = img_path.joinpath(str(i))
        instance_path.mkdir(exist_ok=True)
        cv2.imwrite(filename=str(instance_path.joinpath("{}.png".format(global_step))), img=(2**8 - 1) * overlay)
