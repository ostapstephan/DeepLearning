#!/bin/python3.6

#Luka Lipovac - Homework #3
#Machine Learning - Chris Curro


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import tensorflow as tf
from tqdm import tqdm

NUM_BATCHES = 50000
BATCH_SIZE = 32
LEARNING_RATE = 0.001

SIZE_L1 = 32
SIZE_L2 = 64
SIZE_DENSE = 1024
SIZE_OUT = 10

IMAGE_SIZE_X = 28
IMAGE_SIZE_Y = 28
MAXPOOLS = 2
VAL_PERC = 15


class Data(object):
    def __init__(self, images_train, labels_train):
        self.images = images_train
        self.labels = labels_train
        self.index_in_epoch = 0
        self.num_examples = images.shape[0]

    def get_batch(self):
        start = self.index_in_epoch
        self.index_in_epoch += BATCH_SIZE

        # When all the training data is ran, shuffles it
        if self.index_in_epoch > self.num_examples:
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.labels = self.labels[perm]
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self.index_in_epoch

        return np.reshape(self.images[start:end], [-1, 28, 28, 1]), self.labels[start:end]


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def split_data(images_train, labels_train, validation_percentage):
    percent_quantity = int(round(validation_percentage/100*len(images_train)))
    return images_train[percent_quantity:], labels_train[percent_quantity:], images_train[:percent_quantity], labels_train[:percent_quantity]

def f(images, keep_prob):
    ##layer definitions
    w_1 = tf.get_variable('w_1', [5, 5, 1, SIZE_L1], tf.float32, tf.random_normal_initializer())
    b_1 = tf.get_variable('b_1', [1, SIZE_L1], tf.float32, tf.zeros_initializer())

    w_2 = tf.get_variable('w_2', [5, 5, SIZE_L1, SIZE_L2], tf.float32, tf.random_normal_initializer())
    b_2 = tf.get_variable('b_2', [1, SIZE_L2], tf.float32, tf.zeros_initializer())
    #dense layer
    maxpooled_size = int((IMAGE_SIZE_X*IMAGE_SIZE_Y)/((2*MAXPOOLS)**2))
    w_dense = tf.get_variable('w3_dense', [maxpooled_size*SIZE_L2, SIZE_DENSE], tf.float32, tf.random_normal_initializer())
    b_dense = tf.get_variable('b3_dense', [1, SIZE_DENSE], tf.float32, tf.zeros_initializer())

    w_output = tf.get_variable('w3_output', [SIZE_DENSE, SIZE_OUT], tf.float32, tf.random_normal_initializer())
    b_output = tf.get_variable('b3_output', [1, SIZE_OUT], tf.float32, tf.zeros_initializer())

    ##connections
    layer_1 = max_pool_2x2(tf.nn.relu(conv2d(images, w_1) + b_1))
    layer_2 = max_pool_2x2(tf.nn.relu(conv2d(layer_1, w_2) + b_2))

    layer_2_flattened = tf.reshape(layer_2, [-1, maxpooled_size*SIZE_L2])
    layer_dense = tf.nn.relu(tf.matmul(layer_2_flattened,w_dense) + b_dense)
    #dropout
    layer_dense_dropout = tf.nn.dropout(layer_dense, keep_prob)

    return (tf.matmul(layer_dense_dropout,w_output)+b_output)

#load and make data usable
(images_train, labels_train), (images_test, labels_test) = tf.keras.datasets.mnist.load_data()
images_train, images_test = images_train/255.0, images_test/255.0
labels_train, labels_test = tf.keras.utils.to_categorical(labels_train), tf.keras.utils.to_categorical(labels_test)
#split data into test and val
images_train, labels_train, images_val, labels_val = split_data(images_train, labels_train, VAL_PERC)

images = tf.placeholder(tf.float32, [None, IMAGE_SIZE_X, IMAGE_SIZE_Y, 1])
labels = tf.placeholder(tf.float32, [None, SIZE_OUT])
keep_prob = tf.placeholder(tf.float32)
images_cnn = f(images, keep_prob)


lam = 0.001
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=images_cnn) + lam*tf.reduce_sum([tf.nn.l2_loss(tV) for tV in tf.trainable_variables()])
optim = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
init = tf.global_variables_initializer()

#import pdb; pdb.set_trace()
#print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

sess = tf.Session()
sess.run(init)

data = Data(images_train, labels_train)
for i in tqdm(range(0, NUM_BATCHES)):
    images_np, labels_np= data.get_batch()
    loss_np, _ = sess.run([loss, optim], feed_dict={images: images_np, labels: labels_np, keep_prob: 0.25})
    print(loss_np)


#Evaluation
correct_prediction = tf.equal(tf.argmax(images_cnn, 1), tf.argmax(labels, 1))
predict = tf.argmax(images_cnn, 1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cv_accuracy = accuracy.eval(session=sess, feed_dict={images: np.reshape(images_val, [-1, 28, 28, 1]), labels: labels_val, keep_prob: 1.0})
print('validation_accuracy => %.4f'%cv_accuracy)
sess.close()
