
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os , re
import sys
from tensorflow.examples.tutorials.mnist import input_data

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
  return np.frombuffer(buffer,
         dtype='u1' if int(maxval) < 256 else byteorder+'u2',
         count=int(width)*int(height),
         offset=len(header)
        ).reshape((int(height)*int(width)))

def import_images(image_dir, num_images):
  images_tensor = np.zeros((num_images, 250*250))
  i = 0
  for dirName, subdirList, fileList in os.walk(image_dir):
    for fname in fileList:
      if fname.endswith(".pgm"):
        images_tensor[i] = read_pgm(image_dir+fname, byteorder='<')
        i += 1

  # Create a tensor for the labels
  labels_tensor = np.zeros(num_images,dtype=np.int32)
  f = open("labels.txt", 'r')
  i=0;

  for line in f:
    image_num = i
    labels_tensor[image_num] = int(line[0]) - 1
    i+=1;
    #print("image "+str(i)+ " saved ");

  out = np.zeros((num_images, 7))
  out[np.arange(num_images), labels_tensor] = 1
  return images_tensor, out

images,labels = import_images("./img/" , 1256)
test_image , test_labels =images[:50],labels[:50]
#sys.exit();

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Input layer
x  = tf.placeholder(tf.float32, [None, 250 * 250], name='x')
y_ = tf.placeholder(tf.float32, [None, 7],  name='y_')
x_image = tf.reshape(x, [-1, 250, 250, 1])

# Convolutional layer 1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Convolutional layer 2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully connected layer 1
h_pool2_flat = tf.reshape(h_pool2, [-1, 63*63*64])

W_fc1 = weight_variable([63 * 63 * 64, 1024])
b_fc1 = bias_variable([1024])
print(h_pool2_flat.shape , W_fc1.shape)

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob  = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully connected layer 2 (Output layer)
W_fc2 = weight_variable([1024, 7])
b_fc2 = bias_variable([7])

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y')

# Evaluation functions
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

# Training algorithm
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
batchindex = 0;
# Training steps
with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  max_steps = 1000
  for step in range(max_steps):
    tempx , tempy = [] , []
    for j in range(50):
      tempx.append(images[batchindex + j])
      tempy.append(labels[batchindex + j])
    batchindex += 50
    batch_xs, batch_ys = tempx, tempy
    if (step % 100) == 0:
      print(step, sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0}))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
    print(max_steps, sess.run(accuracy, feed_dict={x: test_image, y_: test_labels, keep_prob: 1.0}))
