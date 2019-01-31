import os
import imageio
from argparse import ArgumentParser
from glob import glob
import numpy as np
import cv2
from skimage.util import montage


def main():
    # Instantiate parser
    parser = ArgumentParser()

    parser.add_argument("--img_dir",
                        help="Where the images reside",
                        required=True)

    parser.add_argument("--batch_indices",
                        nargs="+",
                        required=True)

    parser.add_argument("--step_size",
                        type=int,
                        default=10)

    parser.add_argument("--horizon",
                        type=int,
                        default=2000)

    # Parse
    args = parser.parse_args()

    # Input assertions
    assert os.path.isdir(args.img_dir), "Directory does not exist!"

    collage = []
    for i, batch_index in enumerate(args.batch_indices):

        # Load images
        image_fnames = glob(os.path.join(args.img_dir, batch_index, "*.png"))
        image_fnames.sort(key=lambda fname: int(os.path.splitext(os.path.basename(fname))[0]))
        images = [imageio.imread(image_fname) for image_fname in image_fnames]

        # Add text overlay
        overlays = []
        for step, image in enumerate(images):
            if step % args.step_size > 0 or step > args.horizon:
                continue

            overlays.append(cv2.putText(image,
                                        "Step: {}".format(step),
                                        (30, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1,
                                        (0, 0, 0),
                                        2) if i == 0 else image)

        # Add to collage
        collage.append(overlays)

    # Merge images
    collage = [montage(np.stack(images_at_step, axis=0), multichannel=True) for images_at_step in zip(*collage)]

    # Write images
    imageio.mimsave(os.path.join(os.path.dirname(args.img_dir), "animation.gif"), collage)


if __name__ == "__main__":
    main()
