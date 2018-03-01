"""
Author: Moustafa Alzantot (malzantot@ucla.edu)
"""

import tensorflow as tf

def reset_graph():
	'''reset graph. if session is on, close it'''
    sess = tf.get_default_session()
    if sess:
        sess.close()
    tf.reset_default_graph()


def fully_connected(input_node, num_outputs, scope=None):
    """ Implements a fully connected layer. """
    w_initializer = tf.truncated_normal_initializer(stddev=0.1)		# FC layer is initialized using truncated normal and constant
    b_initializer = tf.constant_initializer(0.0)
    input_shape = input_node.get_shape()
    with tf.variable_scope(scope or 'fc'):
        w = tf.get_variable(name='w', shape=[input_shape[1], num_outputs], initializer=w_initializer)
        b = tf.get_variable(name='b', shape=[num_outputs], initializer=b_initializer)
    fc = tf.matmul(input_node, w) + b
    return fc, (w, b)


def conv_layer(input_node, num_filters, filter_sz, stride=1, scope=None):
    """ Implements a convolution layer. """
    w_initializer = tf.truncated_normal_initializer(stddev=0.1)		# Conv layer is initialized using truncated normal and constant(or random_normal)
    b_initializer = tf.constant_initializer(0.0)
    input_shape = input_node.get_shape()  # input_shape supposed to be [batch, h, w, d]
    with tf.variable_scope(scope or 'conv'):
        w = tf.get_variable(name='w', shape=[filter_sz, filter_sz, input_shape[-1], num_filters], initializer=w_initializer)
        b = tf.get_variable(name='b', shape=[num_filters])
        conv_out = tf.nn.conv2d(input_node, w, strides=[1, stride, stride, 1], padding='SAME')
        #conv_out = conv_out + b
        conv_out = tf.nn.bias_add(conv_out, b)
    return conv_out, (w, b)
    

def train_model(session,
                train_model,
                test_model,
                train_x,
                train_y,
                test_x,
                test_y,
                n_epochs=1000,
                verbose=True,
                log_every=50,
                test_every=100):
    for i in range(n_epochs):
        epoch_cost, epoch_accuracy = train_model.train_epoch(session, train_x, train_y)
        if verbose and i % log_every == 0:
            print('Epoch %d - Cost : %0.4f , Accuracy = %0.2f' %((i+1), epoch_cost, epoch_accuracy))
    
        if verbose and i % test_every == 0:
            test_accuracy = test_model.evaluate_accuracy(session, test_x, test_y)
            print('Epoch: %d - Test accuracy = %0.2f' %((i+1), test_accuracy))