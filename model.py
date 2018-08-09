import tensorflow as tf
from architecture import shufflenet


MOMENTUM = 0.9
USE_NESTEROV = False


def model_fn(features, labels, mode, params):
    """
    This is a function for creating a computational tensorflow graph.
    The function is in format required by tf.estimator.
    """

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    logits = shufflenet(
        features['images'], num_classes=params['num_classes'],
        depth_multiplier=params['depth_multiplier']
    )

    if not is_training:
        predictions = {
            'probabilities': tf.nn.softmax(logits, axis=1),
            'classes': tf.reduce_max(logits, axis=1)
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

    with tf.name_scope('cross_entropy_loss'):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels['labels'], logits=logits)
        tf.losses.add_loss(loss)

    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
    tf.summary.scalar('cross_entropy_loss', loss)
    tf.summary.scalar('regularization_loss', regularization_loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels['labels'], predictions['classes'])
        }
        return tf.estimator.EstimatorSpec(
            mode, loss=total_loss,
            eval_metric_ops=eval_metric_ops
        )

    assert mode == tf.estimator.ModeKeys.TRAIN
    with tf.variable_scope('learning_rate'):
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.polynomial_decay(
            params['initial_learning_rate'], global_step,
            params['decay_steps'], params['end_learning_rate'],
            power=1.0
        )  # linear decay
        tf.summary.scalar('learning_rate', learning_rate)

    with tf.variable_scope('optimizer'):
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM, use_nesterov=USE_NESTEROV)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        minimize_op = optimizer.apply_gradients(grads_and_vars, global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)

    for g, v in grads_and_vars:
        tf.summary.histogram(v.name[:-2] + '_hist', v)
        tf.summary.histogram(v.name[:-2] + '_grad_hist', g)

    with tf.name_scope('evaluation_ops'):
        predicted_labels = tf.argmax(logits, axis=1, output_type=tf.int32)
        train_accuracy = tf.reduce_mean(tf.to_float(tf.equal(labels['labels'], predicted_labels)), axis=0)
    tf.summary.scalar('train_accuracy', train_accuracy)

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)


def add_weight_decay(weight_decay):
    weights = [
        v for v in tf.trainable_variables()
        if 'weights' in v.name and 'depthwise_weights' not in v.name
    ]
    for w in weights:
        value = tf.multiply(weight_decay, tf.nn.l2_loss(w))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, value)
