"""
Image classification Models.

Author: Moustafa Alzantot(malzantot@ucla.edu)
"""
import numpy as np
import tensorflow as tf


import model_utils

class SoftmaxModel(object):
    def __init__(self, is_training=False, learning_rate=0.1, batch_size = 128):
        self.learning_rate = learning_rate
        self.is_training = is_training
        self.batch_size = batch_size
        self._build_model()
    
    def _build_model(self):
        with tf.name_scope('softmax_model'):
            self.x_holder = tf.placeholder(tf.float32, [None, 784])
            self.y_holder = tf.placeholder(tf.float32, [None, 10])
            
            self.fc1, (self.w, self.b) = model_utils.fully_connected(self.x_holder, 10, scope='fc')
            self.logits = self.fc1
            self.preds = tf.nn.softmax(self.logits)
            self.correct_preds = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y_holder, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_preds, tf.float32))
            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y_holder,
                                                        logits=self.logits)
            )
            self.cost_grad_to_input = tf.gradients(self.cost, self.x_holder)
            
            if self.is_training:

                self.opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                self.train_op = self.opt.minimize(self.cost)
                
    def train_epoch(self, session, train_x, train_y):
        assert self.is_training, "Must be a training model."
        assert train_x.shape[1] == 784, "Wrong shape for training data."
        assert train_y.shape[1] == 10, "Wrong shape for training labels."
        batches_per_epoch = train_x.shape[0] // self.batch_size
        epoch_cost = 0.
        epoch_accuracy = 0.
        fetch_list = [self.train_op, self.cost, self.accuracy]
        for batch_idx in range(batches_per_epoch):
            batch_xs = train_x[batch_idx * self.batch_size: (batch_idx+1)*self.batch_size]
            batch_ys = train_y[batch_idx * self.batch_size: (batch_idx+1)*self.batch_size]
                
            feed_dict ={
                self.x_holder: batch_xs,
                self.y_holder: batch_ys
            }
            _, batch_cost , batch_accuracy = session.run(fetch_list, feed_dict=feed_dict)
            epoch_cost += batch_cost
            epoch_accuracy += batch_accuracy
        epoch_cost = epoch_cost / batches_per_epoch
        epoch_accuracy = 100. * epoch_accuracy / batches_per_epoch
        return epoch_cost, epoch_accuracy
    
    def evaluate_accuracy(self, session, test_x, test_y):
        assert test_x.shape[1] == 784, "Wrong shape for testing data."
        assert test_y.shape[1] == 10, "Wrong shape for testing labels."
        test_accuracy = session.run(
            self.accuracy, feed_dict = {
                self.x_holder: test_x,
                self.y_holder: test_y
            }
        )
        return test_accuracy
    
    def predict_label(self, session, test_x):
        assert test_x.shape[1] == 784
        pred_logits = session.run(self.logits, feed_dict = {
            self.x_holder: test_x
        })
        pred_results = np.argmax(pred_logits, axis=1)
        return pred_results
    
    def compute_cost_to_input_gradient(self, session, data_x, data_y):
        assert data_x.shape[1] == 784, "Wrong shape for input data."
        assert data_y.shape[1] == 10, "Wrong shape for input labels."
        grad_val = session.run(
            self.cost_grad_to_input,
            feed_dict = {
                self.x_holder: data_x,
                self.y_holder: data_y
            }
        )
        return grad_val[0]

   
class MLPModel(object):
    def __init__(self, is_training=False, n_hidden=512, learning_rate=0.1, batch_size = 128):
        self.learning_rate = learning_rate
        self.is_training = is_training
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self._build_model()
    
    def _build_model(self):
        with tf.name_scope('mlp_model'):
            self.x_holder = tf.placeholder(tf.float32, [None, 784])
            self.y_holder = tf.placeholder(tf.float32, [None, 10])
            
            self.fc1, (self.w1, self.b1) = model_utils.fully_connected(self.x_holder, self.n_hidden, scope='fc1')
            self.fc1 = tf.nn.relu(self.fc1)
            
            self.fc2, (self.w2, self.b2) = model_utils.fully_connected(self.fc1, 10, scope='fc2')
            self.logits = self.fc2
            self.preds = tf.nn.softmax(self.logits)
            self.correct_preds = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y_holder, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_preds, tf.float32))
            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y_holder,
                                                        logits=self.logits)
            )
            self.cost_grad_to_input = tf.gradients(self.cost, self.x_holder)
            
            if self.is_training:

                self.opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                self.train_op = self.opt.minimize(self.cost)
                
    def train_epoch(self, session, train_x, train_y):
        assert self.is_training, "Must be a training model."
        assert train_x.shape[1] == 784, "Wrong shape for training data."
        assert train_y.shape[1] == 10, "Wrong shape for training labels."
        batches_per_epoch = train_x.shape[0] // self.batch_size
        epoch_cost = 0.
        epoch_accuracy = 0.
        fetch_list = [self.train_op, self.cost, self.accuracy]
        for batch_idx in range(batches_per_epoch):
            batch_xs = train_x[batch_idx * self.batch_size: (batch_idx+1)*self.batch_size]
            batch_ys = train_y[batch_idx * self.batch_size: (batch_idx+1)*self.batch_size]
                
            feed_dict ={
                self.x_holder: batch_xs,
                self.y_holder: batch_ys
            }
            _, batch_cost , batch_accuracy = session.run(fetch_list, feed_dict=feed_dict)
            epoch_cost += batch_cost
            epoch_accuracy += batch_accuracy
        epoch_cost = epoch_cost / batches_per_epoch
        epoch_accuracy = 100. * epoch_accuracy / batches_per_epoch
        return epoch_cost, epoch_accuracy
    
    def evaluate_accuracy(self, session, test_x, test_y):
        assert test_x.shape[1] == 784, "Wrong shape for testing data."
        assert test_y.shape[1] == 10, "Wrong shape for testing labels."
        test_accuracy = session.run(
            self.accuracy, feed_dict = {
                self.x_holder: test_x,
                self.y_holder: test_y
            }
        )
        return test_accuracy
    
    def predict_label(self, session, test_x):
        assert test_x.shape[1] == 784
        pred_logits = session.run(self.logits, feed_dict = {
            self.x_holder: test_x
        })
        pred_results = np.argmax(pred_logits, axis=1)
        return pred_results
    
    def compute_cost_to_input_gradient(self, session, data_x, data_y):
        assert data_x.shape[1] == 784, "Wrong shape for input data."
        assert data_y.shape[1] == 10, "Wrong shape for input labels."
        grad_val = session.run(
            self.cost_grad_to_input,
            feed_dict = {
                self.x_holder: data_x,
                self.y_holder: data_y
            }
        )
        return grad_val[0]

class CNNModel(object):
    def __init__(self, is_training=False, learning_rate=0.1, batch_size = 128, input_node=None):
        self.learning_rate = learning_rate
        self.is_training = is_training
        self.batch_size = batch_size
        self.input_node = input_node
        self._build_model()
    
    def _build_model(self):
        with tf.name_scope('cnn_model'):
            if self.input_node is not None:
                self.x_holder = self.input_node
            else:
                self.x_holder = tf.placeholder(tf.float32, [None, 28, 28, 1])
             
            self.y_holder = tf.placeholder(tf.float32, [None, 10])
            
            with tf.name_scope('conv1'):
                self.conv1, (self.w1, self.b1) = model_utils.conv_layer(self.x_holder, 4, 3, 1, 'conv1')
                self.conv1 = tf.nn.relu(self.conv1)
                self.mp1 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1], strides=[1,2, 2,1], padding='SAME')
            
            with tf.name_scope('conv2'):
                self.conv2, (self.w2, self.b2) = model_utils.conv_layer(self.mp1, 16, 3, 1, 'conv2')
                self.conv2 = tf.nn.relu(self.conv2)
                self.mp2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2,1], padding='SAME')

            mp_shape = self.mp2.get_shape()
            flat_dim = int(mp_shape[1] * mp_shape[2] * mp_shape[3])
            self.flat_conv = tf.reshape(self.mp2, [-1, flat_dim])
            
            with tf.name_scope('fc'):
                self.fc3, (self.w3, self.b3) = model_utils.fully_connected(self.flat_conv, 10, scope='fc3')
            
            self.logits = self.fc3
            self.preds = tf.nn.softmax(self.logits)
            self.correct_preds = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y_holder, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_preds, tf.float32))
            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y_holder,
                                                        logits=self.logits)
            )
            self.cost_grad_to_input = tf.gradients(self.cost, self.x_holder)
            
            if self.is_training:

                self.opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                self.train_op = self.opt.minimize(self.cost)
                
    def train_epoch(self, session, train_x, train_y):
        assert self.is_training, "Must be a training model."
        #assert train_x.shape[1] == 784, "Wrong shape for training data."
        #assert train_y.shape[1] == 10, "Wrong shape for training labels."
        batches_per_epoch = train_x.shape[0] // self.batch_size
        epoch_cost = 0.
        epoch_accuracy = 0.
        fetch_list = [self.train_op, self.cost, self.accuracy]
        for batch_idx in range(batches_per_epoch):
            batch_xs = train_x[batch_idx * self.batch_size: (batch_idx+1)*self.batch_size]
            batch_ys = train_y[batch_idx * self.batch_size: (batch_idx+1)*self.batch_size]
                
            feed_dict ={
                self.x_holder: batch_xs,
                self.y_holder: batch_ys
            }
            _, batch_cost , batch_accuracy = session.run(fetch_list, feed_dict=feed_dict)
            epoch_cost += batch_cost
            epoch_accuracy += batch_accuracy
        epoch_cost = epoch_cost / batches_per_epoch
        epoch_accuracy = 100. * epoch_accuracy / batches_per_epoch
        return epoch_cost, epoch_accuracy
    
    def evaluate_accuracy(self, session, test_x, test_y):
        #assert test_x.shape[1] == 784, "Wrong shape for testing data."
        #assert test_y.shape[1] == 10, "Wrong shape for testing labels."
        test_accuracy = session.run(
            self.accuracy, feed_dict = {
                self.x_holder: test_x,
                self.y_holder: test_y
            }
        )
        return test_accuracy
    
    def predict_label(self, session, test_x):
        #assert test_x.shape[1] == 784
        pred_logits = session.run(self.logits, feed_dict = {
            self.x_holder: test_x
        })
        pred_results = np.argmax(pred_logits, axis=1)
        return pred_results
    
    def compute_cost_to_input_gradient(self, session, data_x, data_y):
        #assert data_x.shape[1] == 784, "Wrong shape for input data."
        #assert data_y.shape[1] == 10, "Wrong shape for input labels."
        grad_val = session.run(
            self.cost_grad_to_input,
            feed_dict = {
                self.x_holder: data_x,
                self.y_holder: data_y
            }
        )
        return grad_val[0]