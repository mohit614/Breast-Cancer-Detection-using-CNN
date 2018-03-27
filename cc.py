from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
import re
import numpy
import os
import tensorflow as tf
from random import randint

def read_pgm(filename, byteorder='>'):
  with open(filename, 'rb') as f:
    buffer = f.read()
  try:
    header, width, height, maxval = re.search(
  b"(^P5\s(?:\s*#.*[\r\n])*"
  b"(\d+)\s(?:\s*#.*[\r\n])*"
  b"(\d+)\s(?:\s*#.*[\r\n])*"
  b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
  except AttributeError:
    raise ValueError("Not a raw PGM file: '%s'" % filename)
  return numpy.frombuffer(buffer,
         dtype='u1' if int(maxval) < 256 else byteorder+'u2',
         count=int(width)*int(height),
         offset=len(header)
        ).reshape((int(height)*int(width)))


def import_images(image_dir, num_images):
  images_tensor = numpy.zeros((num_images, 250*250))
  i = 0
  for dirName, subdirList, fileList in os.walk(image_dir):
    for fname in fileList:
      if fname.endswith(".pgm"):
        images_tensor[i] = read_pgm(image_dir+fname, byteorder='<')
        i += 1

  # Create a tensor for the labels
  labels_tensor = numpy.zeros(num_images,dtype=np.int32)
  f = open("labels.txt", 'r')
  i=0;

  for line in f:
    image_num = i
    labels_tensor[image_num] = int(line[0])
    i+=1;
    print("image "+str(i)+ " saved ");

  return images_tensor, labels_tensor


def cnn_model_fn(features, labels, mode):
  input_layer = tf.reshape(features["x"], [-1, 250, 250, 1])

 
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2 ], strides=2)
  print(pool1.shape)

  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[11, 11],
      padding="same",
      activation=tf.nn.relu)
  
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  print(pool2.shape)
 
  pool2_flat = tf.reshape(pool2, [-1, 62 *62* 64])

  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  logits = tf.layers.dense(inputs=dropout, units=7)

  predictions = {
      
      "classes": tf.argmax(input=logits, axis=1),
     
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  
  train_data  , train_labels = import_images("./img/",1256)

  eval_data , eval_labels = train_data  , train_labels
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="./tmp/breast_cancer")

  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data.astype(np.float32)},
      y=train_labels.astype(np.int32),
      batch_size=10,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=1000,
      hooks=[logging_hook])

  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data.astype(np.float32)},
      y=eval_labels.astype(np.int32),
      batch_size=5,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

if __name__ == "__main__":
  tf.app.run()

