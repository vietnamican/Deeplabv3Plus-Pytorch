import glob
import os.path
import numpy as np
import cv2

from PIL import Image

import tensorflow as tf

FLAGS = tf.compat.v1.flags.FLAGS

tf.compat.v1.flags.DEFINE_string('original_gt_folder',
                                 './VOCdevkit/VOC2012/SegmentationClass',
                                 'Original ground truth annotations.')

tf.compat.v1.flags.DEFINE_string('segmentation_format', 'png', 'Segmentation format.')

tf.compat.v1.flags.DEFINE_string('output_dir',
                                 './VOCdevkit/VOC2012/SegmentationClassRaw',
                                 'folder to save modified ground truth annotations.')


def _remove_colormap(filename):
    """Removes the color map from the annotation.
    Args:
      filename: Ground truth annotation filename.
    Returns:
      Annotation without color map.
      Read file in palette mode in Pil Image, 1 byte for 1 pixel - index in the palette
    """
    return np.array(Image.open(filename))


def _save_annotation(annotation, filename):
    """Saves the annotation as png file.
    Args:
      annotation: Segmentation annotation.
      filename: Output filename.
    """
    pil_image = Image.fromarray(annotation.astype(dtype=np.uint8))
    with tf.io.gfile.GFile(filename, mode='w') as f:
        pil_image.save(f, 'PNG')


def main(unused_argv):
    # Create the output directory if not exists.
    if not tf.io.gfile.isdir(FLAGS.output_dir):
        tf.io.gfile.makedirs(FLAGS.output_dir)

    annotations = glob.glob(os.path.join(FLAGS.original_gt_folder,
                                         '*.' + FLAGS.segmentation_format))
    for annotation in annotations:
        raw_annotation = _remove_colormap(annotation)
        filename = os.path.basename(annotation)[:-4]
        _save_annotation(raw_annotation,
                         os.path.join(
                             FLAGS.output_dir,
                             filename + '.' + FLAGS.segmentation_format))


if __name__ == '__main__':
    _remove_colormap("./pascal_voc_seg/VOCdevkit/VOC2012/SegmentationClass/2007_000032.png")
# tf.compat.v1.app.run()
