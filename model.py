import tensorflow as tf
from architecture import shufflenet


MOMENTUM = 0.9
USE_NESTEROV = True
MOVING_AVERAGE_DECAY = 0.995


def model_fn(features, labels, mode, params):
    """
    This is a function for creating a computational tensorflow graph.
    The function is in format required by tf.estimator.
    """

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    logits = shufflenet(
        features['images'], is_training,
        num_classes=params['num_classes'],
        depth_multiplier=params['depth_multiplier']
    )
    predictions = {
        'probabilities': tf.nn.softmax(logits, axis=1),
        'classes': tf.argmax(logits, axis=1, output_type=tf.int32)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = tf.estimator.export.PredictOutput({
            name: tf.identity(tensor, name)
            for name, tensor in predictions.items()
        })
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions,
            export_outputs={'outputs': export_outputs}
        )

    with tf.name_scope('weight_decay'):
        add_weight_decay(params['weight_decay'])
        regularization_loss = tf.losses.get_regularization_loss()

    with tf.name_scope('cross_entropy'):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels['labels'], logits=logits)
        loss = tf.reduce_mean(losses, axis=0)
        tf.losses.add_loss(loss)

    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
    tf.summary.scalar('cross_entropy_loss', loss)
    tf.summary.scalar('regularization_loss', regularization_loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels['labels'], predictions['classes']),
            'top5_accuracy': tf.metrics.mean(tf.to_float(tf.nn.in_top_k(
                predictions=predictions['probabilities'], targets=labels['labels'], k=5
            )))
        }
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, eval_metric_ops=eval_metric_ops)

    assert mode == tf.estimator.ModeKeys.TRAIN
    with tf.variable_scope('learning_rate'):
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.polynomial_decay(
            params['initial_learning_rate'], global_step,
            params['decay_steps'], params['end_learning_rate'],
            power=1.0
        )  # linear decay
        tf.summary.scalar('learning_rate', learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops), tf.variable_scope('optimizer'):
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM, use_nesterov=USE_NESTEROV)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    for g, v in grads_and_vars:
        tf.summary.histogram(v.name[:-2] + '_hist', v)
        tf.summary.histogram(v.name[:-2] + '_grad_hist', g)

    with tf.name_scope('evaluation_ops'):
        train_accuracy = tf.reduce_mean(tf.to_float(tf.equal(
            labels['labels'], predictions['classes']
        )), axis=0)
        train_top5_accuracy = tf.reduce_mean(tf.to_float(tf.nn.in_top_k(
            predictions=predictions['probabilities'], targets=labels['labels'], k=5
        )), axis=0)
    tf.summary.scalar('train_accuracy', train_accuracy)
    tf.summary.scalar('train_top5_accuracy', train_top5_accuracy)

    with tf.control_dependencies([train_op]), tf.name_scope('ema'):
        ema = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY, num_updates=global_step)
        train_op = ema.apply(tf.trainable_variables())

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)


def add_weight_decay(weight_decay):
    weights = [
        v for v in tf.trainable_variables()
        if 'weights' in v.name and 'depthwise_weights' not in v.name
    ]
    for w in weights:
        value = tf.multiply(weight_decay, tf.nn.l2_loss(w))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, value)


class RestoreMovingAverageHook(tf.train.SessionRunHook):
    def __init__(self, model_dir):
        super(RestoreMovingAverageHook, self).__init__()
        self.model_dir = model_dir

    def begin(self):
        ema = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY)
        variables_to_restore = ema.variables_to_restore()
        self.load_ema = tf.contrib.framework.assign_from_checkpoint_fn(
            tf.train.latest_checkpoint(self.model_dir), variables_to_restore
        )

    def after_create_session(self, sess, coord):
        tf.logging.info('Loading EMA weights...')
        self.load_ema(sess)
