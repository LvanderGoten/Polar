import numpy as np
import cv2

from polar import gen


def get_test_batch(batch_size, width, height, circle_radius, device):
    return next(gen(batch_size=batch_size,
                    width=width,
                    height=height,
                    circle_radius=circle_radius,
                    device=device))


def draw_test_batch(image_batch, radius_batch, phi_batch, summary_writer, global_step, width, height, circle_radius):
    # Re-order image dimensions
    image_batch = np.transpose(image_batch, axes=[0, 2, 3, 1])

    # Spatial
    center_x = width // 2
    center_y = height // 2
    max_radial_dist = np.sqrt(center_x ** 2 + center_y ** 2)

    for i, (image, radius, phi) in enumerate(zip(image_batch, radius_batch, phi_batch)):
        # Cartesian coordinates
        x = (max_radial_dist * radius * np.cos(phi)).astype(np.int32)
        y = (max_radial_dist * radius * np.sin(phi)).astype(np.int32)

        # Draw prediction
        overlay = cv2.circle(img=image.copy(),
                             center=(center_x + x, center_y - y),
                             radius=circle_radius,
                             color=(0, 0, 0))

        # Re-order image dimensions
        overlay = np.transpose(overlay, axes=[2, 0, 1])

        # TensorBoard
        summary_writer.add_image(tag="prediction_{}".format(i),
                                 img_tensor=overlay,
                                 global_step=global_step)
