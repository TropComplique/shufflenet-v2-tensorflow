import tensorflow as tf
import tensorflow.contrib.slim as slim


BATCH_NORM_MOMENTUM = 0.997
BATCH_NORM_EPSILON = 1e-3


def shufflenet(images, is_training, num_classes=1000, depth_multiplier='1.0'):
    """
    This is an implementation of ShuffleNet v2:
    https://arxiv.org/abs/1807.11164

    Arguments:
        images: a float tensor with shape [batch_size, image_height, image_width, 3],
            a batch of RGB images with pixel values in the range [0, 1].
        is_training: a boolean.
        num_classes: an integer.
        depth_multiplier: a string, possible values are '0.5', '1.0', '1.5', and '2.0'.
    Returns:
        a float tensor with shape [batch_size, num_classes].
    """
    possibilities = {'0.5': 48, '1.0': 116, '1.5': 176, '2.0': 224}
    initial_depth = possibilities[depth_multiplier]

    def batch_norm(x):
        x = tf.layers.batch_normalization(
            x, axis=3, center=True, scale=True,
            training=is_training,
            momentum=BATCH_NORM_MOMENTUM,
            epsilon=BATCH_NORM_EPSILON,
            fused=True, name='batch_norm'
        )
        return x

    with tf.name_scope('standardize_input'):
        x = (2.0 * images) - 1.0

    with tf.variable_scope('ShuffleNetV2'):
        params = {
            'padding': 'SAME', 'activation_fn': tf.nn.relu,
            'normalizer_fn': batch_norm, 'data_format': 'NHWC',
            'weights_initializer': tf.contrib.layers.xavier_initializer()
        }
        with slim.arg_scope([slim.conv2d, depthwise_conv], **params):

            x = slim.conv2d(x, 24, (3, 3), stride=2, scope='Conv1')
            x = slim.max_pool2d(x, (3, 3), stride=2, padding='SAME', scope='MaxPool')

            x = block(x, num_units=4, out_channels=initial_depth, scope='Stage2')
            x = block(x, num_units=8, scope='Stage3')
            x = block(x, num_units=4, scope='Stage4')

            final_channels = 1024 if depth_multiplier != '2.0' else 2048
            x = slim.conv2d(x, final_channels, (1, 1), stride=1, scope='Conv5')

    # global average pooling
    x = tf.reduce_mean(x, axis=[1, 2])

    logits = slim.fully_connected(
        x, num_classes, activation_fn=None, scope='classifier',
        weights_initializer=tf.contrib.layers.xavier_initializer()
    )
    return logits


def block(x, num_units, out_channels=None, scope='stage'):
    with tf.variable_scope(scope):

        with tf.variable_scope('unit_1'):
            x, y = basic_unit_with_downsampling(x, out_channels)

        for j in range(2, num_units + 1):
            with tf.variable_scope('unit_%d' % j):
                x, y = concat_shuffle_split(x, y)
                x = basic_unit(x)
        x = tf.concat([x, y], axis=3)

    return x


def concat_shuffle_split(x, y):
    with tf.name_scope('concat_shuffle_split'):
        shape = tf.shape(x)
        batch_size = shape[0]
        height, width = shape[1], shape[2]
        depth = x.shape[3].value

        z = tf.stack([x, y], axis=3)  # shape [batch_size, height, width, 2, depth]
        z = tf.transpose(z, [0, 1, 2, 4, 3])
        z = tf.reshape(z, [batch_size, height, width, 2*depth])
        x, y = tf.split(z, num_or_size_splits=2, axis=3)
        return x, y


def basic_unit(x):
    in_channels = x.shape[3].value
    x = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_before')
    x = depthwise_conv(x, kernel=3, stride=1, activation_fn=None, scope='depthwise')
    x = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_after')
    return x


def basic_unit_with_downsampling(x, out_channels=None):
    in_channels = x.shape[3].value
    out_channels = 2 * in_channels if out_channels is None else out_channels

    y = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_before')
    y = depthwise_conv(y, kernel=3, stride=2, activation_fn=None, scope='depthwise')
    y = slim.conv2d(y, out_channels // 2, (1, 1), stride=1, scope='conv1x1_after')

    with tf.variable_scope('second_branch'):
        x = depthwise_conv(x, kernel=3, stride=2, activation_fn=None, scope='depthwise')
        x = slim.conv2d(x, out_channels // 2, (1, 1), stride=1, scope='conv1x1_after')
        return x, y


@tf.contrib.framework.add_arg_scope
def depthwise_conv(
        x, kernel=3, stride=1, padding='SAME',
        activation_fn=None, normalizer_fn=None,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        data_format='NHWC', scope='depthwise_conv'):

    with tf.variable_scope(scope):
        assert data_format == 'NHWC'
        in_channels = x.shape[3].value
        W = tf.get_variable(
            'depthwise_weights',
            [kernel, kernel, in_channels, 1], dtype=tf.float32,
            initializer=weights_initializer
        )
        x = tf.nn.depthwise_conv2d(x, W, [1, stride, stride, 1], padding, data_format='NHWC')
        x = normalizer_fn(x) if normalizer_fn is not None else x  # batch normalization
        x = activation_fn(x) if activation_fn is not None else x  # nonlinearity
        return x
