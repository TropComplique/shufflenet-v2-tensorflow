import tensorflow as tf


SHUFFLE_BUFFER_SIZE = 10000
NUM_FILES_READ_IN_PARALLEL = 10
NUM_PARALLEL_CALLS = 8
RESIZE_METHOD = tf.image.ResizeMethod.BILINEAR
IMAGE_SIZE = 224  # this will be used for training and evaluation
MIN_DIMENSION = 256


class Pipeline:
    def __init__(self, filenames, is_training, batch_size, num_epochs, num_gpus):
        """
        Arguments:
            filenames: a list of strings, paths to tfrecords files.
            is_training: a boolean.
            batch_size, num_epochs, num_gpus: integers.
        """
        self.is_training = is_training

        # get the number of images in the dataset
        def get_num_samples(filename):
            return sum(1 for _ in tf.python_io.tf_record_iterator(filename))
        num_examples = 0
        for filename in filenames:
            num_examples_in_file = get_num_samples(filename)
            assert num_examples_in_file > 0
            num_examples += num_examples_in_file
        assert num_examples > 0
        self.num_examples = num_examples

        # read the files in parallel
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        num_shards = len(filenames)
        if is_training:
            dataset = dataset.shuffle(buffer_size=num_shards)
        dataset = dataset.apply(tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=NUM_FILES_READ_IN_PARALLEL
        ))
        dataset = dataset.prefetch(buffer_size=batch_size)

        # mix the training examples
        if is_training:
            dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        dataset = dataset.repeat(num_epochs)

        # force the number of batches to be divisible by the number of devices
        if is_training and num_gpus > 1:
            total_examples = num_epochs * num_examples
            total_batches = ((total_examples // batch_size) // num_gpus) * num_gpus
            dataset = dataset.take(total_batches * batch_size)

        # decode and augment data
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            self.parse_and_preprocess, batch_size=batch_size,
            num_parallel_batches=1, drop_remainder=False
        ))
        dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

        self.dataset = dataset

    def parse_and_preprocess(self, example_proto):
        """What this function does:
        1. Parses one record from a tfrecords file and decodes it.
        2. Possibly augments it.

        Returns:
            image: a float tensor with shape [height, width, 3],
                a RGB image with pixel values in the range [0, 1].
            label: an int tensor with shape [].
        """
        features = {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'ymin': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'xmin': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'ymax': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'xmax': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
        }
        parsed_features = tf.parse_single_example(example_proto, features)

        # get an image as a string, it will be decoded later
        image_as_string = parsed_features['image']

        # get a label
        label = tf.to_int32(parsed_features['label'])

        # get groundtruth boxes, they must be in from-zero-to-one format,
        # also, it assumed that ymin < ymax and xmin < xmax
        boxes = tf.stack([
            parsed_features['ymin'], parsed_features['xmin'],
            parsed_features['ymax'], parsed_features['xmax']
        ], axis=1)
        boxes = tf.to_float(boxes)  # shape [num_boxes, 4]

        if self.is_training:
            image = self.augmentation(image_as_string, boxes)
        else:
            image = tf.image.decode_jpeg(image_as_string, channels=3)
            image = (1.0 / 255.0) * tf.to_float(image)
            image = resize_keeping_aspect_ratio(image, MIN_DIMENSION)
            image = central_crop(image, crop_height=IMAGE_SIZE, crop_width=IMAGE_SIZE)

        image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])

        # in the format required by tf.estimator,
        # they will be batched later
        features = {'images': image}
        labels = {'labels': label}
        return features, labels

    def augmentation(self, image_as_string, boxes):

        image = get_random_crop(image_as_string, boxes)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.resize_images(
            image, [IMAGE_SIZE, IMAGE_SIZE],
            method=RESIZE_METHOD
        )
        image = (1.0 / 255.0) * tf.to_float(image)

        # note that color augmentations are very slow!
        image = random_color_manipulations(image, probability=0.05, grayscale_probability=0.01)
        return image


def resize_keeping_aspect_ratio(image, min_dimension):
    """
    Arguments:
        image: a float tensor with shape [height, width, 3].
        min_dimension: an int tensor with shape [].
    Returns:
        a float tensor with shape [new_height, new_width, 3],
            where min_dimension = min(new_height, new_width).
    """
    image_shape = tf.shape(image)
    height = tf.to_float(image_shape[0])
    width = tf.to_float(image_shape[1])

    original_min_dim = tf.minimum(height, width)
    scale_factor = tf.to_float(min_dimension) / original_min_dim
    new_height = tf.round(height * scale_factor)
    new_width = tf.round(width * scale_factor)

    new_size = [tf.to_int32(new_height), tf.to_int32(new_width)]
    image = tf.image.resize_images(image, new_size, method=RESIZE_METHOD)
    return image


def get_random_crop(image_as_string, boxes):

    distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.image.extract_jpeg_shape(image_as_string),
        bounding_boxes=tf.expand_dims(boxes, axis=0),
        min_object_covered=0.5,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.1, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True
    )
    begin, size, _ = distorted_bounding_box
    offset_y, offset_x, _ = tf.unstack(begin)
    target_height, target_width, _ = tf.unstack(size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

    crop = tf.image.decode_and_crop_jpeg(
        image_as_string, crop_window, channels=3
    )
    return crop


def central_crop(image, crop_height, crop_width):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2

    return tf.slice(image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])


def random_color_manipulations(image, probability=0.1, grayscale_probability=0.1):

    def manipulate(image):
        # intensity and order of this operations are kinda random,
        # so you will need to tune this for you problem
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_hue(image, 0.1)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    def to_grayscale(image):
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.grayscale_to_rgb(image)
        return image

    with tf.name_scope('random_color_manipulations'):
        do_it = tf.less(tf.random_uniform([]), probability)
        image = tf.cond(do_it, lambda: manipulate(image), lambda: image)

    with tf.name_scope('to_grayscale'):
        make_gray = tf.less(tf.random_uniform([]), grayscale_probability)
        image = tf.cond(make_gray, lambda: to_grayscale(image), lambda: image)

    return image


# def distort_color_fast(image, scope=None):
#   with tf.name_scope(scope, 'distort_color', [image]):
#     br_delta = random_ops.random_uniform([], -32./255., 32./255., seed=None)
#     cb_factor = random_ops.random_uniform(
#         [], -FLAGS.cb_distortion_range, FLAGS.cb_distortion_range, seed=None)
#     cr_factor = random_ops.random_uniform(
#         [], -FLAGS.cr_distortion_range, FLAGS.cr_distortion_range, seed=None)

#     channels = tf.split(axis=2, num_or_size_splits=3, value=image)
#     red_offset = 1.402 * cr_factor + br_delta
#     green_offset = -0.344136 * cb_factor - 0.714136 * cr_factor + br_delta
#     blue_offset = 1.772 * cb_factor + br_delta
#     channels[0] += red_offset
#     channels[1] += green_offset
#     channels[2] += blue_offset
#     image = tf.concat(axis=2, values=channels)
#     image = tf.clip_by_value(image, 0., 1.)
#     return image
