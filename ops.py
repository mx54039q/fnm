#coding:utf-8
import math
import numpy as np 
import tensorflow as tf
from config import cfg

#def bn(x,is_train=True):
#    return tf.layers.batch_normalization(x,epsilon=1e-5,momentum=0.9,
#        training=is_train)
class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name
  def __call__(self, x, is_train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=is_train,
                      scope=self.name)

def local(x,filters,name,kernel_size=3,strides=[1,1],padding='valid'):
    with tf.variable_scope(name):
        return tf.contrib.keras.layers.LocallyConnected2D(
                 filters=filters,
                 kernel_size=kernel_size,
                 strides=strides,
                 padding=padding,
                 kernel_initializer=tf.truncated_normal_initializer(stddev=cfg.stddev))(x)

def conv2d(inputs, filters, name, kernel_size = 3, strides = [1,1], padding='same',
           dilation_rate = 1, trainable = True, activation = None, reuse = False):
    return tf.layers.conv2d(inputs, filters = filters,
             kernel_size = kernel_size,
             padding = padding,
             strides = strides,
             dilation_rate = dilation_rate,
             activation = activation,
             trainable = trainable,
             reuse = reuse,
             kernel_initializer = tf.truncated_normal_initializer(stddev=cfg.stddev),
             kernel_regularizer = tf.contrib.layers.l2_regularizer(0.0001),
             name = name)

def deconv2d(inputs, filters, name, kernel_size = 3, strides = [1,1], padding='same',
           trainable = True, activation = None, reuse = False):
    return tf.layers.conv2d_transpose(inputs, filters = filters,
             kernel_size = kernel_size,
             padding = padding,
             strides = strides,
             activation = activation,
             trainable = trainable,
             reuse = reuse,
             kernel_initializer = tf.truncated_normal_initializer(stddev=cfg.stddev),
             kernel_regularizer = tf.contrib.layers.l2_regularizer(0.0001),
             name = name)

def deconv2d_w(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def fullyConnect(inputs, units, name, trainable = True, activation = None, reuse = False):
    return tf.layers.dense(inputs, units = units,
             kernel_initializer = tf.truncated_normal_initializer(stddev=cfg.stddev),
             kernel_regularizer = tf.contrib.layers.l2_regularizer(0.0001),
             activation = activation,
             trainable = trainable,
             reuse = reuse,
             name = name)
             
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)                      
