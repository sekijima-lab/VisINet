"""Implementation reference: https://github.com/horovod/horovod/examples ."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd
import socket
import os
import time
import ratio
import resnet_v2
import sys
import random
slim = tf.contrib.slim
learn = tf.contrib.learn
argvs = sys.argv
pos_weight = eval("ratio." + "sirt1")()
print(pos_weight)
repeats = 30
batch_size = 128

tf.logging.set_verbosity(tf.logging.INFO)

def enrichmentfactor(scores, label, ratio=0.01):
   num_picked = int(scores.size * ratio)
   random_hitrate = np.sum(label == 1) / label.size
   order = np.argsort(scores)[::-1]
   top_label = label[order][:num_picked]
   predicted_hitrate = np.sum(top_label == 1) / num_picked
   return predicted_hitrate / random_hitrate



def parser(serialized_example):
  """Parses a single tf.Example into image and label tensors."""
  features = tf.parse_single_example(
      serialized_example,
      features={
          'image': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })
  image = tf.decode_raw(features['image'], tf.float32)
  label = features['label']
  #train_label = tf.decode_raw(train_exam['label'], tf.int64)
  image = tf.reshape(image,[150528])
  image = tf.cast(image, tf.float32) * (1. / 255)
  image = tf.reshape(image, [224, 224, 3])
  label = tf.one_hot(label, 2)
  label = tf.cast(label,tf.float32)
  label = tf.reshape(label,[2])
  return image, label

def crop():
    k = random.randint(1, 4)
    print(k)
    image =  tf.image.rot90(image,k=k)
    label = label
    return image, label

# Define the input function for training
def train_input_fn():
  #iterator = dataset.make_one_shot_iterator()
  #features, labels = iterator.get_next()
  #files = tf.data.Dataset.list_files("/gs/hs0/tga-science/img_dock/tf/" + argvs[2] + "_224_train.tfrecord")
  #dataset = files.interleave(tf.data.TFRecordDataset,cycle_length=28)
  #dataset = dataset.shuffle(10000 + 3 * batch_size)
  #dataset = dataset.map(map_func=parser)
  #features, labels = dataset.batch(batch_size=batch_size)
  '''
  dataset = tf.data.TFRecordDataset("/gs/hs0/tga-science/img_dock/tf/" + argvs[2] + "_224_train.tfrecord")
  dataset = dataset.map(parser)
  dataset = dataset.repeat(4)
  dataset = dataset.shuffle(10000 + 3 * batch_size)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()
  '''
  #d = tf.data.Dataset.list_files("/gs/hs0/tga-science/img_dock/tf/" + argvs[2] + "_train_*.tfrecord")
  d = tf.data.TFRecordDataset(argvs[2:])
  d = d.shard(hvd.size(), hvd.rank())
  #d = d.interleave(tf.data.TFRecordDataset, cycle_length = 4, block_length=2)
  d = d.repeat(repeats)
  d = d.shuffle(10000)
  d = d.map(parser, num_parallel_calls=28)
  d = d.batch(batch_size)
  iterator = d.make_one_shot_iterator()
  features, labels = iterator.get_next()
  k = random.randint(1, 4)
  features = tf.map_fn(lambda x: tf.image.rot90(x,k=k), features)

  return features, labels


def eval_input_fn():
  dataset = tf.data.TFRecordDataset(argvs[2:])
  dataset = dataset.map(parser)
        # eval_dataset = eval_dataset.repeat(FLAGS.num_epochs)
  dataset = dataset.batch(81)
  iterator = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()
  return features, labels


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    x_image = tf.reshape(features, [-1, 224, 224, 3])
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        nets, end_points = resnet_v2.resnet_v2_50(x_image, num_classes = 2, is_training = False)
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(nets, 1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(end_points['predictions'], name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=nets, pos_weight = pos_weight))

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Horovod: scale learning rate by the number of workers.
        optimizer = tf.train.AdamOptimizer(1e-4)

        # Horovod: add Horovod Distributed Optimizer.
        optimizer = hvd.DistributedOptimizer(optimizer)

        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        #"true_label": tf.argmax(labels, 1),
        #"classes": tf.argmax(nets, 1),
        "accuracy": tf.metrics.accuracy(
            labels=tf.argmax(labels, 1), predictions=tf.argmax(nets, 1)),
        "AUC": tf.metrics.auc(labels=tf.argmax(labels, 1), predictions=tf.argmax(nets, 1))}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Horovod: initialize Horovod.
    hvd.init()

    # Load training and eval data

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting them.
    model_dir = argvs[1] if hvd.rank() == 0 else None

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=model_dir,
        config=tf.estimator.RunConfig(session_config=config))

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, 
        every_n_iter=1000)

    # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states from
    # rank 0 to all other processes. This is necessary to ensure consistent
    # initialization of all workers when training is started with random weights or
    # restored from a checkpoint.
    bcast_hook = hvd.BroadcastGlobalVariablesHook(0)


    # Train the model
    # Horovod: adjust number of steps based on number of GPUs.
    mnist_classifier.train(
        input_fn=train_input_fn,
        #steps=200 // hvd.size(),
        hooks=[logging_hook, bcast_hook])

    # Evaluate the model and print results
    predictions = list(mnist_classifier.predict(input_fn=eval_input_fn))
    print(len(predictions))
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
