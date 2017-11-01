from __future__ import print_function
import numpy as np
import os
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

data_root = 'C:/data'
num_labels = 10
image_size = 28
num_channels = 1  # grayscale

# 10000 -> 0.001 l2 -> 90 95.2  with 256 hidden Max Pool 91.4 -> 96.3


with open(os.path.join(data_root, 'notMNIST.pickle'), 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(preds, labels):
  return 100. * np.sum(np.argmax(preds,axis=1)==np.argmax(labels,axis=1))/preds.shape[0]

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


batch_size = 128
patch_size = 5
depth = 16
num_hidden = 256 
beta = 0.001

graph = tf.Graph()
with graph.as_default():

  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  layer1_weights = tf.get_variable("layer1_weights", shape=[patch_size, patch_size, num_channels, depth], initializer=tf.contrib.layers.xavier_initializer())
  layer1_biases = tf.Variable(tf.zeros([depth]))
  layer2_weights = tf.get_variable("layer2_weights", shape=[patch_size, patch_size, depth, depth], initializer=tf.contrib.layers.xavier_initializer())
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
  layer3_weights = tf.get_variable("layer3_weights", shape=[image_size // 4 * image_size // 4 * depth, num_hidden], initializer=tf.contrib.layers.xavier_initializer())
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
  layer4_weights = tf.get_variable("layer4_weights", shape=[num_hidden, num_labels], initializer=tf.contrib.layers.xavier_initializer())
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
  # Model.
  def model(data):
    conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
    # pooling after relu
    hidden = tf.nn.max_pool(tf.nn.relu(conv + layer1_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print(hidden.shape)
    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer2_biases)  
    #hidden = tf.nn.max_pool(tf.nn.relu(conv + layer2_biases), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    print(hidden.shape)
    return tf.matmul(hidden, layer4_weights) + layer4_biases
  
  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  loss = loss + beta * tf.nn.l2_loss(layer1_weights) + beta * tf.nn.l2_loss(layer2_weights) + beta * tf.nn.l2_loss(layer3_weights) + beta * tf.nn.l2_loss(layer4_weights)
    
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  #optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))

num_steps = 10001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))