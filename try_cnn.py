'''
Tianwei: standard CNN in tensorflow.
Data loaded from Keras
'''

import numpy as np
import matplotlib

from matplotlib import pyplot as plt
from matplotlib import cm as cmap

import tensorflow as tf

import keras
from keras.datasets import cifar10, mnist
from keras.utils import np_utils

(train_x, train_y), (test_x, test_y) = mnist.load_data()        # original data x: 60k*28*28, y 60k,
train_x = train_x.reshape((-1, 28, 28, 1)).astype(np.float32) / 255.0       # increase x dim, and normalize
batch_size = 200
learning_rate = 0.1
train_y = np_utils.to_categorical(train_y, 10)      # change y to one-hot
test_y = np_utils.to_categorical(test_y, 10)

tf.reset_default_graph()        # Clears the default graph stack and resets the global default graph. 不可在session active时候用
with tf.Graph().as_default():   # set graph: where to store the following nodes and connections
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    y = tf.placeholder(tf.int32, shape=[None, 10])
    
    w1 = tf.Variable(tf.random_normal([3, 3, 1, 32]))
    b1 = tf.Variable(tf.random_normal([32]))
    conv1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.bias_add(conv1, b1)
    conv1 = tf.nn.relu(conv1)
    #mp1 = tf.nn.max_pool(conv1, ksize=[1,2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    #print(mp1.get_shape())
    
    w2 = tf.Variable(tf.random_normal([3, 3, 32, 64]))
    b2 = tf.Variable(tf.random_normal([64]))
    conv2 = tf.nn.conv2d(conv1, w2, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.bias_add(conv2, b2)
    conv2 = tf.nn.relu(conv2)
    mp2 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print(mp2.get_shape())
    flat1 = tf.reshape(mp2, [-1, 14*14* 32])
    
    w3 = tf.get_variable(name='w3', dtype=tf.float32, shape=[14*14*32,1024], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b3 = tf.get_variable(name='b3', dtype=tf.float32, shape=[1024], initializer=tf.constant_initializer())
    fc1 = tf.nn.bias_add(tf.matmul(flat1, w3), b3)
    fc1 = tf.nn.relu(fc1)
    
    w4 = tf.get_variable(name='w4', dtype=tf.float32, shape=[1024, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b4 = tf.get_variable(name='b4', dtype=tf.float32, shape=[10], initializer=tf.constant_initializer())
    
    logits = tf.nn.bias_add(tf.matmul(fc1, w4), b4) # softmax on the result of final layer.
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)) #softmax cross-entropy定义loss
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1)), tf.float32))  #accuracy
    opt = tf.train.GradientDescentOptimizer(learning_rate)  #opt use vanilla SGD
    train_op = opt.minimize(cost)  # training operation is: use opt to minimize cost
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)  # use only 50% of memory
#     self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    vali_cost =[]
    vali_acc = []
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:   # everything runs in this session
#     with tf.Session() as session:
        session.run(tf.global_variables_initializer())  # initialize all variables(trainable weights)
        for k in range(20):                             # train for xx epochs
            for i in range(train_x.shape[0]//batch_size):  # iterate for trainNO/batchSize times(not necessarily visit every sample once)
                batch_idx = np.random.choice(train_x.shape[0], batch_size, replace=False)  # select batch_size from train_num (no-replacement: no repetitive)
                batch_xs = train_x[batch_idx]
                batch_ys = train_y[batch_idx]
                _, cost_val, acc_val = session.run([train_op, cost, accuracy], feed_dict = {x: batch_xs,
                                                                y: batch_ys})
                
            vali_cost_i, vali_acc_i = session.run([cost, accuracy], feed_dict = {x: test_x, y: test_y})
            print('No:%d  Cost:%.6e  Acc:%.6e'%(k, vali_cost_i, vali_acc_i))
            vali_cost.append(vali_cost_i)
            vali_acc.append(vali_acc_i)


def plot_LearningCurve(vali_cost, vali_acc):
    width = 8
    height = 6
    fig= plt.figure(figsize=(width, height))
    ax = fig.add_subplot(1,1,1) #add_subplot(1,1,1)

    plt.plot(np.arange(len(vali_cost)), vali_cost, 'b-.', markersize=1,label= 'Validation Cost')
    plt.plot(np.arange(len(vali_acc)), vali_acc, 'r-o', markersize=3, label= 'Validation Error-rate')

    ax.grid()
    ax.legend()
    plt.xlabel('Num of Epochs')
    plt.ylabel('Validation Accuracy and Cost')
    plt.show()

plot_LearningCurve(vali_cost, vali_acc)
