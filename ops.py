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
             kernel_regularizer = tf.contrib.layers.l2_regularizer(0.00001),
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
             kernel_regularizer = tf.contrib.layers.l2_regularizer(0.00001),
             name = name)

def fullyConnect(inputs, units, name, trainable = True, activation = None, reuse = False):
    return tf.layers.dense(inputs, units = units,
             kernel_initializer = tf.truncated_normal_initializer(stddev=cfg.stddev),
             kernel_regularizer = tf.contrib.layers.l2_regularizer(0.00001),
             activation = activation,
             trainable = trainable,
             reuse = reuse,
             name = name)             
