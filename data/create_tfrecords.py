import io
import os
from PIL import Image
import tensorflow as tf
import json
import numpy as np
import shutil
import random
import pandas as pd
import math
import argparse
from tqdm import tqdm


"""
The purpose of this script is to create a set of .tfrecords files
from a folder of images and a folder of annotations.

Labels text file contains a list of all label names.
One label name per line. Number of line - label integer encoding.

Example of use:
python create_tfrecords.py \
    --metadata_file=training.csv \
    --output=/mnt/datasets/imagenet/train_shards/ \
    --labels=integer_encoding.txt \
    --num_shards=100
"""


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--metadata_file', type=str)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-l', '--labels', type=str)
    parser.add_argument('-b', '--boxes', type=str, default='')
    parser.add_argument('-s', '--num_shards', type=int, default=1)
    return parser.parse_args()


def dict_to_tf_example(image_path, integer_label, boxes=None):
    """
    Arguments:
        image_path: a string.
        integer_label: an integer.
        boxes: a numpy float array with shape [num_boxes, 4] or None,
            boxes are in normalized coordinates.
    Returns:
        an instance of tf.Example or None.
    """
    assert image_path.endswith('.JPEG')
    with tf.gfile.GFile(image_path, 'rb') as f:
        encoded_jpg = f.read()

    # check image format
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    if image.mode == 'L':  # if gray
        rgb_image = np.stack(3*[np.array(image)], axis=2)
        encoded_jpg = to_jpeg_bytes(rgb_image)
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
    elif image.mode != 'RGB':
        return None
    if image.format != 'JPEG':
        return None
    assert image.mode == 'RGB'

    assert image.size[0] > 1 and image.size[1] > 1
    assert (0 <= integer_label) and (integer_label <= 999)

    feature = {
        'image': _bytes_feature(encoded_jpg),
        'label': _int64_feature(integer_label)
    }

    if boxes is not None:
        xmin_list, ymin_list, xmax_list, ymax_list = [], [], [], []
        for box in boxes:

            xmin, ymin, xmax, ymax = box

            assert (xmin < xmax) and (ymin < ymax)
            assert (xmin <= 1.0) and (xmin >= 0.0)
            assert (xmax <= 1.0) and (xmax >= 0.0)
            assert (ymin <= 1.0) and (ymin >= 0.0)
            assert (ymax <= 1.0) and (ymax >= 0.0)

            xmin_list.append(xmin)
            ymin_list.append(ymin)
            xmax_list.append(xmax)
            ymax_list.append(ymax)

        feature.update({
            'xmin': _float_list_feature(xmin_list),
            'ymin': _float_list_feature(ymin_list),
            'xmax': _float_list_feature(xmax_list),
            'ymax': _float_list_feature(ymax_list)
        })

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def to_jpeg_bytes(array):
    image = Image.fromarray(array)
    tmp = io.BytesIO()
    image.save(tmp, format='jpeg')
    return tmp.getvalue()


def main():
    ARGS = make_args()

    with open(ARGS.labels, 'r') as f:
        label_encoder = json.load(f)
    assert len(label_encoder) > 0
    print('Number of classes:', len(label_encoder))

    metadata = pd.read_csv(ARGS.metadata_file)
    metadata = metadata.sample(frac=1)  # shuffle images
    print('Number of images:', len(metadata))

    num_shards = ARGS.num_shards
    num_examples = len(metadata)
    shard_size = math.ceil(num_examples/num_shards)
    print('Number of images per shard:', shard_size)
    
    bounding_boxes = None
    if len(ARGS.boxes) > 0:
        bounding_boxes = np.load(ARGS.boxes)[()]
        print('Number of images with boxes:', len(bounding_boxes))

    output_dir = ARGS.output
    shutil.rmtree(output_dir, ignore_errors=True)
    os.mkdir(output_dir)

    shard_id = 0
    num_examples_written = 0
    num_skipped_images = 0
    for T in tqdm(metadata.itertuples()):

        if num_examples_written == 0:
            shard_path = os.path.join(output_dir, 'shard-%04d.tfrecords' % shard_id)
            writer = tf.python_io.TFRecordWriter(shard_path)

        image_path = T.path  # absolute path to an image
        integer_label = label_encoder[T.wordnet_id]
        boxes = None
        if bounding_boxes is not None:
            boxes = bounding_boxes.get(T.just_name, np.empty((0, 4), dtype='float32'))

        tf_example = dict_to_tf_example(image_path, integer_label, boxes)
        if tf_example is None:
            num_skipped_images += 1
            continue
        writer.write(tf_example.SerializeToString())
        num_examples_written += 1

        if num_examples_written == shard_size:
            shard_id += 1
            num_examples_written = 0
            writer.close()

    # this happens if num_examples % num_shards != 0
    if num_examples_written != 0:
        writer.close()
    
    print('Number of skipped images:', num_skipped_images)
    print('Result is here:', ARGS.output)


main()
