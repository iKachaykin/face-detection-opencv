from argparse import ArgumentParser
import os
import numpy as np

import cv2
from tqdm import tqdm

from models.face_detector import FaceDetector
from utils import extract_paths_by_extensions

IMAGE_EXTENSIONS = ('jpeg', 'jpg', 'png',)

parser = ArgumentParser('Detect, cut and resize faces on given images')
parser.add_argument(
    '--source', type=str, required=True,
    help='Path to input images directory'
)
parser.add_argument(
    '--dest', type=str, required=True,
    help='Path to output images directory'
)
parser.add_argument(
    '--shape', nargs=2, default=(256, 256), type=int,
    help='Images output shape'
)


def main(source, dest, shape):
    detector = __create_face_detector()
    images_paths = extract_paths_by_extensions(source, IMAGE_EXTENSIONS)
    images = [cv2.imread(single_path) for single_path in images_paths]
    for single_image in tqdm(images):
        _, boxes, _ = detector.detect(np.expand_dims(single_image, 0))
        orig_shape = __get_image_shape(single_image)
        for single_box in boxes:
            xmin, ymin, xmax, ymax = __extract_limits(single_box, orig_shape)
            single_face = single_image[ymin:ymax, xmin:xmax]
            single_face = cv2.resize(single_face, shape)
            single_path = os.path.join(dest, f'{hash(np.random.rand())}.jpg')
            cv2.imwrite(single_path, single_face)


def __create_face_detector():
    prototxt = os.path.join('resources', 'weights', 'deploy.prototxt')
    weights = os.path.join(
        'resources', 'weights', 'res10_300x300_ssd_iter_140000.caffemodel'
    )
    detector = FaceDetector(
        prototxt=prototxt,
        weights=weights,
        input_size=300,
        confidence_threshold=0.5,
        input_scale=1.0,
        input_swap_rb=False,
        use_gpu=False,
        extract_best=False
    )
    return detector


def __get_image_shape(image):
    h, w = image.shape[:2]
    return w, h


def __extract_limits(
        single_box, original_shape, ymin_scale=0.2, ymax_scale=0.05
):
    xmin, ymin, xmax, ymax = __project_box(single_box, original_shape)
    height = int(ymax - ymin)
    ymin = np.maximum(ymin - ymin_scale*height, 0)
    ymax = int(ymax + ymax_scale*height)
    xmin, ymin, xmax, ymax = __project_box(
        (xmin, ymin, xmax, ymax), original_shape
    )
    width = int(xmax - xmin)
    height = int(ymax - ymin)
    if height > width:
        diff = height - width
        xmin = xmin - int(diff / 2)
        xmax = xmax + int(diff / 2)
    else:
        diff = width - height
        ymin = ymin - int(diff / 2)
        ymax = ymax + int(diff / 2)
    xmin, ymin, xmax, ymax = __project_box(
        (xmin, ymin, xmax, ymax), original_shape
    )
    return xmin, ymin, xmax, ymax


def __project_box(box, original_shape):
    xmin, ymin, xmax, ymax = box
    w, h = original_shape
    xmin = int(np.maximum(xmin, 0))
    ymin = int(np.maximum(ymin, 0))
    xmax = int(np.minimum(xmax, w))
    ymax = int(np.minimum(ymax, h))
    return xmin, ymin, xmax, ymax


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.source, args.dest, tuple(args.shape))
