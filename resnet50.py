# coding: utf-8
# --------------------------------------------------------
# FNM
# Written by Yichen Qian
# --------------------------------------------------------

import os
import numpy as np
import tensorflow as tf
from ops import *

VGG_MEAN = [131.0912, 103.8827, 91.4953] # for channel BGR
#VGG_MEAN = [129.1836, 104.7624, 93.5940] # for channel BGR


class Resnet50(object):
  """Class for Resnet50 model trained on VGGFace2 dataset
  
  This class is for resnet50 model trained on VGGFace2 dataset. Restore 
  pretrained model from binary file. Function "forward" can extract feature 
  from a input face.
  """
  
  def __init__(self, resnet50_npy_path=None):
    self.data_dict = np.load(cfg.face_model, encoding='latin1').item()
    print("npy file loaded")
  
  def build(self):
    """Load parameters from dict and build up the face recognition model
    """
    with tf.variable_scope('resnet50_parameters'):
      # BatchNorm Init
      self.bn1 = batch_norm_mosv(self.data_dict['conv1_7x7_s2_bn'], 'bn1')
      
      self.bn2_1_reduce = batch_norm_mosv(self.data_dict['conv2_1_1x1_reduce_bn'], 'bn2_1_reduce')
      self.bn2_1_3x3 = batch_norm_mosv(self.data_dict['conv2_1_3x3_bn'], 'bn2_1_3x3')
      self.bn2_1_increase = batch_norm_mosv(self.data_dict['conv2_1_1x1_increase_bn'], 'bn2_1_increase')
      self.bn2_1_proj = batch_norm_mosv(self.data_dict['conv2_1_1x1_proj_bn'], 'bn2_1_proj')
      
      self.bn2_2_reduce = batch_norm_mosv(self.data_dict['conv2_2_1x1_reduce_bn'], 'bn2_2_reduce')
      self.bn2_2_3x3 = batch_norm_mosv(self.data_dict['conv2_2_3x3_bn'], 'bn2_2_3x3')
      self.bn2_2_increase = batch_norm_mosv(self.data_dict['conv2_2_1x1_increase_bn'], 'bn2_2_increase')
      
      self.bn2_3_reduce = batch_norm_mosv(self.data_dict['conv2_3_1x1_reduce_bn'], 'bn2_3_reduce')
      self.bn2_3_3x3 = batch_norm_mosv(self.data_dict['conv2_3_3x3_bn'], 'bn2_3_3x3')
      self.bn2_3_increase = batch_norm_mosv(self.data_dict['conv2_3_1x1_increase_bn'], 'bn2_3_increase')
      
      self.bn3_1_reduce = batch_norm_mosv(self.data_dict['conv3_1_1x1_reduce_bn'], 'bn3_1_reduce')
      self.bn3_1_3x3 = batch_norm_mosv(self.data_dict['conv3_1_3x3_bn'], 'bn3_1_3x3')
      self.bn3_1_increase = batch_norm_mosv(self.data_dict['conv3_1_1x1_increase_bn'], 'bn3_1_increase')
      self.bn3_1_proj = batch_norm_mosv(self.data_dict['conv3_1_1x1_proj_bn'], 'bn3_1_proj')
      
      self.bn3_2_reduce = batch_norm_mosv(self.data_dict['conv3_2_1x1_reduce_bn'], 'bn3_2_reduce')
      self.bn3_2_3x3 = batch_norm_mosv(self.data_dict['conv3_2_3x3_bn'], 'bn3_2_3x3')
      self.bn3_2_increase = batch_norm_mosv(self.data_dict['conv3_2_1x1_increase_bn'], 'bn3_2_increase')
      
      self.bn3_3_reduce = batch_norm_mosv(self.data_dict['conv3_3_1x1_reduce_bn'], 'bn3_3_reduce')
      self.bn3_3_3x3 = batch_norm_mosv(self.data_dict['conv3_3_3x3_bn'], 'bn3_3_3x3')
      self.bn3_3_increase = batch_norm_mosv(self.data_dict['conv3_3_1x1_increase_bn'], 'bn3_3_increase')
      
      self.bn3_4_reduce = batch_norm_mosv(self.data_dict['conv3_4_1x1_reduce_bn'], 'bn3_4_reduce')
      self.bn3_4_3x3 = batch_norm_mosv(self.data_dict['conv3_4_3x3_bn'], 'bn3_4_3x3')
      self.bn3_4_increase = batch_norm_mosv(self.data_dict['conv3_4_1x1_increase_bn'], 'bn3_4_increase')
      
      self.bn4_1_reduce = batch_norm_mosv(self.data_dict['conv4_1_1x1_reduce_bn'], 'bn4_1_reduce')
      self.bn4_1_3x3 = batch_norm_mosv(self.data_dict['conv4_1_3x3_bn'], 'bn4_1_3x3')
      self.bn4_1_increase = batch_norm_mosv(self.data_dict['conv4_1_1x1_increase_bn'], 'bn4_1_increase')
      self.bn4_1_proj = batch_norm_mosv(self.data_dict['conv4_1_1x1_proj_bn'], 'bn4_1_proj')
      
      self.bn4_2_reduce = batch_norm_mosv(self.data_dict['conv4_2_1x1_reduce_bn'], 'bn4_2_reduce')
      self.bn4_2_3x3 = batch_norm_mosv(self.data_dict['conv4_2_3x3_bn'], 'bn4_2_3x3')
      self.bn4_2_increase = batch_norm_mosv(self.data_dict['conv4_2_1x1_increase_bn'], 'bn4_2_increase')
      
      self.bn4_3_reduce = batch_norm_mosv(self.data_dict['conv4_3_1x1_reduce_bn'], 'bn4_3_reduce')
      self.bn4_3_3x3 = batch_norm_mosv(self.data_dict['conv4_3_3x3_bn'], 'bn4_3_3x3')
      self.bn4_3_increase = batch_norm_mosv(self.data_dict['conv4_3_1x1_increase_bn'], 'bn4_3_increase')
      
      self.bn4_4_reduce = batch_norm_mosv(self.data_dict['conv4_4_1x1_reduce_bn'], 'bn4_4_reduce')
      self.bn4_4_3x3 = batch_norm_mosv(self.data_dict['conv4_4_3x3_bn'], 'bn4_4_3x3')
      self.bn4_4_increase = batch_norm_mosv(self.data_dict['conv4_4_1x1_increase_bn'], 'bn4_4_increase')
      
      self.bn4_5_reduce = batch_norm_mosv(self.data_dict['conv4_5_1x1_reduce_bn'], 'bn4_5_reduce')
      self.bn4_5_3x3 = batch_norm_mosv(self.data_dict['conv4_5_3x3_bn'], 'bn4_5_3x3')
      self.bn4_5_increase = batch_norm_mosv(self.data_dict['conv4_5_1x1_increase_bn'], 'bn4_5_increase')
      
      self.bn4_6_reduce = batch_norm_mosv(self.data_dict['conv4_6_1x1_reduce_bn'], 'bn4_6_reduce')
      self.bn4_6_3x3 = batch_norm_mosv(self.data_dict['conv4_6_3x3_bn'], 'bn4_6_3x3')
      self.bn4_6_increase = batch_norm_mosv(self.data_dict['conv4_6_1x1_increase_bn'], 'bn4_6_increase')
      
      self.bn5_1_reduce = batch_norm_mosv(self.data_dict['conv5_1_1x1_reduce_bn'], 'bn5_1_reduce')
      self.bn5_1_3x3 = batch_norm_mosv(self.data_dict['conv5_1_3x3_bn'], 'bn5_1_3x3')
      self.bn5_1_increase = batch_norm_mosv(self.data_dict['conv5_1_1x1_increase_bn'], 'bn5_1_increase')
      self.bn5_1_proj = batch_norm_mosv(self.data_dict['conv5_1_1x1_proj_bn'], 'bn5_1_proj')
      
      self.bn5_2_reduce = batch_norm_mosv(self.data_dict['conv5_2_1x1_reduce_bn'], 'bn5_2_reduce')
      self.bn5_2_3x3 = batch_norm_mosv(self.data_dict['conv5_2_3x3_bn'], 'bn5_2_3x3')
      self.bn5_2_increase = batch_norm_mosv(self.data_dict['conv5_2_1x1_increase_bn'], 'bn5_2_increase')
      
      self.bn5_3_reduce = batch_norm_mosv(self.data_dict['conv5_3_1x1_reduce_bn'], 'bn5_3_reduce')
      self.bn5_3_3x3 = batch_norm_mosv(self.data_dict['conv5_3_3x3_bn'], 'bn5_3_3x3')
      self.bn5_3_increase = batch_norm_mosv(self.data_dict['conv5_3_1x1_increase_bn'], 'bn5_3_increase')
      
      # Convolution Layers Weights
      with tf.variable_scope('conv1_7x7_s2'):
        self.conv1_7x7_s2_weights = self.get_filter('conv1_7x7_s2')
        
      with tf.variable_scope('conv2_1_1x1_reduce'):
        self.conv2_1_1x1_reduce_weights = self.get_filter('conv2_1_1x1_reduce')
      with tf.variable_scope('conv2_1_3x3'):
        self.conv2_1_3x3_weights = self.get_filter('conv2_1_3x3')
      with tf.variable_scope('conv2_1_1x1_increase'):
        self.conv2_1_1x1_increase_weights = self.get_filter('conv2_1_1x1_increase')
      with tf.variable_scope('conv2_1_1x1_proj'):
        self.conv2_1_1x1_proj_weights = self.get_filter('conv2_1_1x1_proj')
      
      with tf.variable_scope('conv2_2_1x1_reduce'):
        self.conv2_2_1x1_reduce_weights = self.get_filter('conv2_2_1x1_reduce')
      with tf.variable_scope('conv2_2_3x3'):
        self.conv2_2_3x3_weights = self.get_filter('conv2_2_3x3')
      with tf.variable_scope('conv2_2_1x1_increase'):
        self.conv2_2_1x1_increase_weights = self.get_filter('conv2_2_1x1_increase')

      with tf.variable_scope('conv2_3_1x1_reduce'):
        self.conv2_3_1x1_reduce_weights = self.get_filter('conv2_3_1x1_reduce')
      with tf.variable_scope('conv2_3_3x3'):
        self.conv2_3_3x3_weights = self.get_filter('conv2_3_3x3')
      with tf.variable_scope('conv2_3_1x1_increase'):
        self.conv2_3_1x1_increase_weights = self.get_filter('conv2_3_1x1_increase')
      
      with tf.variable_scope('conv3_1_1x1_reduce'):
        self.conv3_1_1x1_reduce_weights = self.get_filter('conv3_1_1x1_reduce')
      with tf.variable_scope('conv3_1_3x3'):
        self.conv3_1_3x3_weights = self.get_filter('conv3_1_3x3')
      with tf.variable_scope('conv3_1_1x1_increase'):
        self.conv3_1_1x1_increase_weights = self.get_filter('conv3_1_1x1_increase')
      with tf.variable_scope('conv3_1_1x1_proj'):
        self.conv3_1_1x1_proj_weights = self.get_filter('conv3_1_1x1_proj')
      
      with tf.variable_scope('conv3_2_1x1_reduce'):
        self.conv3_2_1x1_reduce_weights = self.get_filter('conv3_2_1x1_reduce')
      with tf.variable_scope('conv3_2_3x3'):
        self.conv3_2_3x3_weights = self.get_filter('conv3_2_3x3')
      with tf.variable_scope('conv3_2_1x1_increase'):
        self.conv3_2_1x1_increase_weights = self.get_filter('conv3_2_1x1_increase')
      
      with tf.variable_scope('conv3_3_1x1_reduce'):
        self.conv3_3_1x1_reduce_weights = self.get_filter('conv3_3_1x1_reduce')
      with tf.variable_scope('conv3_3_3x3'):
        self.conv3_3_3x3_weights = self.get_filter('conv3_3_3x3')
      with tf.variable_scope('conv3_3_1x1_increase'):
        self.conv3_3_1x1_increase_weights = self.get_filter('conv3_3_1x1_increase')
      
      with tf.variable_scope('conv3_4_1x1_reduce'):
        self.conv3_4_1x1_reduce_weights = self.get_filter('conv3_4_1x1_reduce')
      with tf.variable_scope('conv3_4_3x3'):
        self.conv3_4_3x3_weights = self.get_filter('conv3_4_3x3')
      with tf.variable_scope('conv3_4_1x1_increase'):
        self.conv3_4_1x1_increase_weights = self.get_filter('conv3_4_1x1_increase')
        
      with tf.variable_scope('conv4_1_1x1_reduce'):
        self.conv4_1_1x1_reduce_weights = self.get_filter('conv4_1_1x1_reduce')
      with tf.variable_scope('conv4_1_3x3'):
        self.conv4_1_3x3_weights = self.get_filter('conv4_1_3x3')
      with tf.variable_scope('conv4_1_1x1_increase'):
        self.conv4_1_1x1_increase_weights = self.get_filter('conv4_1_1x1_increase')
      with tf.variable_scope('conv4_1_1x1_proj'):
        self.conv4_1_1x1_proj_weights = self.get_filter('conv4_1_1x1_proj')
        
      with tf.variable_scope('conv4_2_1x1_reduce'):
        self.conv4_2_1x1_reduce_weights = self.get_filter('conv4_2_1x1_reduce')
      with tf.variable_scope('conv4_2_3x3'):
        self.conv4_2_3x3_weights = self.get_filter('conv4_2_3x3')
      with tf.variable_scope('conv4_2_1x1_increase'):
        self.conv4_2_1x1_increase_weights = self.get_filter('conv4_2_1x1_increase')
      
      with tf.variable_scope('conv4_3_1x1_reduce'):
        self.conv4_3_1x1_reduce_weights = self.get_filter('conv4_3_1x1_reduce')
      with tf.variable_scope('conv4_3_3x3'):
        self.conv4_3_3x3_weights = self.get_filter('conv4_3_3x3')
      with tf.variable_scope('conv4_3_1x1_increase'):
        self.conv4_3_1x1_increase_weights = self.get_filter('conv4_3_1x1_increase')
      
      with tf.variable_scope('conv4_4_1x1_reduce'):
        self.conv4_4_1x1_reduce_weights = self.get_filter('conv4_4_1x1_reduce')
      with tf.variable_scope('conv4_4_3x3'):
        self.conv4_4_3x3_weights = self.get_filter('conv4_4_3x3')
      with tf.variable_scope('conv4_4_1x1_increase'):
        self.conv4_4_1x1_increase_weights = self.get_filter('conv4_4_1x1_increase')
          
      with tf.variable_scope('conv4_5_1x1_reduce'):
        self.conv4_5_1x1_reduce_weights = self.get_filter('conv4_5_1x1_reduce')
      with tf.variable_scope('conv4_5_3x3'):
        self.conv4_5_3x3_weights = self.get_filter('conv4_5_3x3')
      with tf.variable_scope('conv4_5_1x1_increase'):
        self.conv4_5_1x1_increase_weights = self.get_filter('conv4_5_1x1_increase')
      
      with tf.variable_scope('conv4_6_1x1_reduce'):
        self.conv4_6_1x1_reduce_weights = self.get_filter('conv4_6_1x1_reduce')
      with tf.variable_scope('conv4_6_3x3'):
        self.conv4_6_3x3_weights = self.get_filter('conv4_6_3x3')
      with tf.variable_scope('conv4_6_1x1_increase'):
        self.conv4_6_1x1_increase_weights = self.get_filter('conv4_6_1x1_increase')
        
      with tf.variable_scope('conv5_1_1x1_reduce'):
        self.conv5_1_1x1_reduce_weights = self.get_filter('conv5_1_1x1_reduce')
      with tf.variable_scope('conv5_1_3x3'):
        self.conv5_1_3x3_weights = self.get_filter('conv5_1_3x3')
      with tf.variable_scope('conv5_1_1x1_increase'):
        self.conv5_1_1x1_increase_weights = self.get_filter('conv5_1_1x1_increase')
      with tf.variable_scope('conv5_1_1x1_proj'):
        self.conv5_1_1x1_proj_weights = self.get_filter('conv5_1_1x1_proj')
        
      with tf.variable_scope('conv5_2_1x1_reduce'):
        self.conv5_2_1x1_reduce_weights = self.get_filter('conv5_2_1x1_reduce')
      with tf.variable_scope('conv5_2_3x3'):
        self.conv5_2_3x3_weights = self.get_filter('conv5_2_3x3')
      with tf.variable_scope('conv5_2_1x1_increase'):
        self.conv5_2_1x1_increase_weights = self.get_filter('conv5_2_1x1_increase')
      
      with tf.variable_scope('conv5_3_1x1_reduce'):
        self.conv5_3_1x1_reduce_weights = self.get_filter('conv5_3_1x1_reduce')
      with tf.variable_scope('conv5_3_3x3'):
        self.conv5_3_3x3_weights = self.get_filter('conv5_3_3x3')
      with tf.variable_scope('conv5_3_1x1_increase'):
        self.conv5_3_1x1_increase_weights = self.get_filter('conv5_3_1x1_increase')
    
    # Clear the model dict
    self.data_dict = None
    
  def forward(self, rgb, scope = 'resnet50'):
    """Forward process of face recognition model
    
    args:
      rgb: rgb image tensors with shape(batch, height, width, 3), values range in [0,255]
    return:
      a set of tensors of layers
    """
    
    with tf.name_scope(scope):
      # Convert RGB to BGR as VGG model do
      assert rgb.get_shape().as_list()[1] == 224
      assert rgb.get_shape().as_list()[2] == 224
      assert rgb.get_shape().as_list()[3] == 3
      red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb)
      bgr = tf.concat(axis=3, values=[
        blue - VGG_MEAN[0],
        green - VGG_MEAN[1],
        red - VGG_MEAN[2],
      ])
      assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
      
      # Construct Model
      with tf.name_scope('conv1'):
        conv1_7x7_s2 = tf.nn.conv2d(bgr, self.conv1_7x7_s2_weights, [1,2,2,1], padding='SAME')
        conv1_7x7_s2 = tf.nn.relu(self.bn1(conv1_7x7_s2))
      # output shape:[112, 112, 64]
      pool1_3x3_s2 = tf.layers.max_pooling2d(conv1_7x7_s2, 3, 2, padding='SAME')
      # output shape:[56, 56, 64]
      with tf.name_scope('conv2_1'):
        conv2_1_proj = tf.nn.conv2d(pool1_3x3_s2, self.conv2_1_1x1_proj_weights, [1,1,1,1], padding='SAME')
        conv2_1_proj = self.bn2_1_proj(conv2_1_proj)
        conv2_1_1x1_reduce = tf.nn.conv2d(pool1_3x3_s2, self.conv2_1_1x1_reduce_weights, [1,1,1,1], padding='SAME')
        conv2_1_1x1_reduce = tf.nn.relu(self.bn2_1_reduce(conv2_1_1x1_reduce))
        conv2_1_3x3 = tf.nn.conv2d(conv2_1_1x1_reduce, self.conv2_1_3x3_weights, [1,1,1,1], padding='SAME')
        conv2_1_3x3 = tf.nn.relu(self.bn2_1_3x3(conv2_1_3x3))
        conv2_1_1x1_increase = tf.nn.conv2d(conv2_1_3x3, self.conv2_1_1x1_increase_weights, [1,1,1,1], padding='SAME')
        conv2_1_1x1_increase = self.bn2_1_increase(conv2_1_1x1_increase)
        conv2_1 = tf.nn.relu(tf.add(conv2_1_1x1_increase, conv2_1_proj))
      # output shape:[56, 56, 256]
      with tf.name_scope('conv2_2'):
        conv2_2_1x1_reduce = tf.nn.conv2d(conv2_1, self.conv2_2_1x1_reduce_weights, [1,1,1,1], padding='SAME')
        conv2_2_1x1_reduce = tf.nn.relu(self.bn2_2_reduce(conv2_2_1x1_reduce))
        conv2_2_3x3 = tf.nn.conv2d(conv2_2_1x1_reduce, self.conv2_2_3x3_weights, [1,1,1,1], padding='SAME')
        conv2_2_3x3 = tf.nn.relu(self.bn2_2_3x3(conv2_2_3x3))
        conv2_2_1x1_increase = tf.nn.conv2d(conv2_2_3x3, self.conv2_2_1x1_increase_weights, [1,1,1,1], padding='SAME')
        conv2_2_1x1_increase = self.bn2_2_increase(conv2_2_1x1_increase)
        conv2_2 = tf.nn.relu(tf.add(conv2_2_1x1_increase, conv2_1))
      # output shape:[56, 56, 256]
      with tf.name_scope('conv2_3'): 
        conv2_3_1x1_reduce = tf.nn.conv2d(conv2_2, self.conv2_3_1x1_reduce_weights, [1,1,1,1], padding='SAME')
        conv2_3_1x1_reduce = tf.nn.relu(self.bn2_3_reduce(conv2_3_1x1_reduce))
        conv2_3_3x3 = tf.nn.conv2d(conv2_3_1x1_reduce, self.conv2_3_3x3_weights, [1,1,1,1], padding='SAME')
        conv2_3_3x3 = tf.nn.relu(self.bn2_3_3x3(conv2_3_3x3))
        conv2_3_1x1_increase = tf.nn.conv2d(conv2_3_3x3, self.conv2_3_1x1_increase_weights, [1,1,1,1], padding='SAME')
        conv2_3_1x1_increase = self.bn2_3_increase(conv2_3_1x1_increase)
        conv2_3 = tf.nn.relu(tf.add(conv2_3_1x1_increase, conv2_2))
      # output shape:[56, 56, 256]
      with tf.name_scope('conv3_1'):
        conv3_1_proj = tf.nn.conv2d(conv2_3, self.conv3_1_1x1_proj_weights, [1,2,2,1], padding='SAME')
        conv3_1_proj = self.bn3_1_proj(conv3_1_proj)
        conv3_1_1x1_reduce = tf.nn.conv2d(conv2_3, self.conv3_1_1x1_reduce_weights, [1,2,2,1], padding='SAME')
        conv3_1_1x1_reduce = tf.nn.relu(self.bn3_1_reduce(conv3_1_1x1_reduce))
        conv3_1_3x3 = tf.nn.conv2d(conv3_1_1x1_reduce, self.conv3_1_3x3_weights, [1,1,1,1], padding='SAME')
        conv3_1_3x3 = tf.nn.relu(self.bn3_1_3x3(conv3_1_3x3))
        conv3_1_1x1_increase = tf.nn.conv2d(conv3_1_3x3, self.conv3_1_1x1_increase_weights, [1,1,1,1], padding='SAME')
        conv3_1_1x1_increase = self.bn3_1_increase(conv3_1_1x1_increase)
        conv3_1 = tf.nn.relu(tf.add(conv3_1_1x1_increase, conv3_1_proj))
      # output shape:[28, 28, 512]
      with tf.name_scope('conv3_2'): 
        conv3_2_1x1_reduce = tf.nn.conv2d(conv3_1, self.conv3_2_1x1_reduce_weights, [1,1,1,1], padding='SAME')
        conv3_2_1x1_reduce = tf.nn.relu(self.bn3_2_reduce(conv3_2_1x1_reduce))
        conv3_2_3x3 = tf.nn.conv2d(conv3_2_1x1_reduce, self.conv3_2_3x3_weights, [1,1,1,1], padding='SAME')
        conv3_2_3x3 = tf.nn.relu(self.bn3_2_3x3(conv3_2_3x3))
        conv3_2_1x1_increase = tf.nn.conv2d(conv3_2_3x3, self.conv3_2_1x1_increase_weights, [1,1,1,1], padding='SAME')
        conv3_2_1x1_increase = self.bn3_2_increase(conv3_2_1x1_increase)
        conv3_2 = tf.nn.relu(tf.add(conv3_2_1x1_increase, conv3_1)) 
      # output shape:[28, 28, 512]
      with tf.name_scope('conv3_3'): 
        conv3_3_1x1_reduce = tf.nn.conv2d(conv3_2, self.conv3_3_1x1_reduce_weights, [1,1,1,1], padding='SAME')
        conv3_3_1x1_reduce = tf.nn.relu(self.bn3_3_reduce(conv3_3_1x1_reduce))
        conv3_3_3x3 = tf.nn.conv2d(conv3_3_1x1_reduce, self.conv3_3_3x3_weights, [1,1,1,1], padding='SAME')
        conv3_3_3x3 = tf.nn.relu(self.bn3_3_3x3(conv3_3_3x3))
        conv3_3_1x1_increase = tf.nn.conv2d(conv3_3_3x3, self.conv3_3_1x1_increase_weights, [1,1,1,1], padding='SAME')
        conv3_3_1x1_increase = self.bn3_3_increase(conv3_3_1x1_increase)
        conv3_3 = tf.nn.relu(tf.add(conv3_3_1x1_increase, conv3_2))
      # output shape:[28, 28, 512]
      with tf.name_scope('conv3_4'): 
        conv3_4_1x1_reduce = tf.nn.conv2d(conv3_3, self.conv3_4_1x1_reduce_weights, [1,1,1,1], padding='SAME')
        conv3_4_1x1_reduce = tf.nn.relu(self.bn3_4_reduce(conv3_4_1x1_reduce))
        conv3_4_3x3 = tf.nn.conv2d(conv3_4_1x1_reduce, self.conv3_4_3x3_weights, [1,1,1,1], padding='SAME')
        conv3_4_3x3 = tf.nn.relu(self.bn3_4_3x3(conv3_4_3x3))
        conv3_4_1x1_increase = tf.nn.conv2d(conv3_4_3x3, self.conv3_4_1x1_increase_weights, [1,1,1,1], padding='SAME')
        conv3_4_1x1_increase = self.bn3_4_increase(conv3_4_1x1_increase)
        conv3_4 = tf.nn.relu(tf.add(conv3_4_1x1_increase, conv3_3))   
      # output shape:[28, 28, 512]
      with tf.name_scope('conv4_1'):
        conv4_1_proj = tf.nn.conv2d(conv3_4, self.conv4_1_1x1_proj_weights, [1,2,2,1], padding='SAME')
        conv4_1_proj = self.bn4_1_proj(conv4_1_proj)
        conv4_1_1x1_reduce = tf.nn.conv2d(conv3_4, self.conv4_1_1x1_reduce_weights, [1,2,2,1], padding='SAME')
        conv4_1_1x1_reduce = tf.nn.relu(self.bn4_1_reduce(conv4_1_1x1_reduce))
        conv4_1_3x3 = tf.nn.conv2d(conv4_1_1x1_reduce, self.conv4_1_3x3_weights, [1,1,1,1], padding='SAME')
        conv4_1_3x3 = tf.nn.relu(self.bn4_1_3x3(conv4_1_3x3))
        conv4_1_1x1_increase = tf.nn.conv2d(conv4_1_3x3, self.conv4_1_1x1_increase_weights, [1,1,1,1], padding='SAME')
        conv4_1_1x1_increase = self.bn4_1_increase(conv4_1_1x1_increase)
        conv4_1 = tf.nn.relu(tf.add(conv4_1_1x1_increase, conv4_1_proj))
      # output shape:[14, 14, 1024]
      with tf.name_scope('conv4_2'): 
        conv4_2_1x1_reduce = tf.nn.conv2d(conv4_1, self.conv4_2_1x1_reduce_weights, [1,1,1,1], padding='SAME')
        conv4_2_1x1_reduce = tf.nn.relu(self.bn4_2_reduce(conv4_2_1x1_reduce))
        conv4_2_3x3 = tf.nn.conv2d(conv4_2_1x1_reduce, self.conv4_2_3x3_weights, [1,1,1,1], padding='SAME')
        conv4_2_3x3 = tf.nn.relu(self.bn4_2_3x3(conv4_2_3x3))
        conv4_2_1x1_increase = tf.nn.conv2d(conv4_2_3x3, self.conv4_2_1x1_increase_weights, [1,1,1,1], padding='SAME')
        conv4_2_1x1_increase = self.bn4_2_increase(conv4_2_1x1_increase)
        conv4_2 = tf.nn.relu(tf.add(conv4_2_1x1_increase, conv4_1)) 
      # output shape:[14, 14, 1024]
      with tf.name_scope('conv4_3'): 
        conv4_3_1x1_reduce = tf.nn.conv2d(conv4_2, self.conv4_3_1x1_reduce_weights, [1,1,1,1], padding='SAME')
        conv4_3_1x1_reduce = tf.nn.relu(self.bn4_3_reduce(conv4_3_1x1_reduce))
        conv4_3_3x3 = tf.nn.conv2d(conv4_3_1x1_reduce, self.conv4_3_3x3_weights, [1,1,1,1], padding='SAME')
        conv4_3_3x3 = tf.nn.relu(self.bn4_3_3x3(conv4_3_3x3))
        conv4_3_1x1_increase = tf.nn.conv2d(conv4_3_3x3, self.conv4_3_1x1_increase_weights, [1,1,1,1], padding='SAME')
        conv4_3_1x1_increase = self.bn4_3_increase(conv4_3_1x1_increase)
        conv4_3 = tf.nn.relu(tf.add(conv4_3_1x1_increase, conv4_2)) 
      # output shape:[14, 14, 1024]
      with tf.name_scope('conv4_4'): 
        conv4_4_1x1_reduce = tf.nn.conv2d(conv4_3, self.conv4_4_1x1_reduce_weights, [1,1,1,1], padding='SAME')
        conv4_4_1x1_reduce = tf.nn.relu(self.bn4_4_reduce(conv4_4_1x1_reduce))
        conv4_4_3x3 = tf.nn.conv2d(conv4_4_1x1_reduce, self.conv4_4_3x3_weights, [1,1,1,1], padding='SAME')
        conv4_4_3x3 = tf.nn.relu(self.bn4_4_3x3(conv4_4_3x3))
        conv4_4_1x1_increase = tf.nn.conv2d(conv4_4_3x3, self.conv4_4_1x1_increase_weights, [1,1,1,1], padding='SAME')
        conv4_4_1x1_increase = self.bn4_4_increase(conv4_4_1x1_increase)
        conv4_4 = tf.nn.relu(tf.add(conv4_4_1x1_increase, conv4_3))
      # output shape:[14, 14, 1024]
      with tf.name_scope('conv4_5'): 
        conv4_5_1x1_reduce = tf.nn.conv2d(conv4_4, self.conv4_5_1x1_reduce_weights, [1,1,1,1], padding='SAME')
        conv4_5_1x1_reduce = tf.nn.relu(self.bn4_5_reduce(conv4_5_1x1_reduce))
        conv4_5_3x3 = tf.nn.conv2d(conv4_5_1x1_reduce, self.conv4_5_3x3_weights, [1,1,1,1], padding='SAME')
        conv4_5_3x3 = tf.nn.relu(self.bn4_5_3x3(conv4_5_3x3))
        conv4_5_1x1_increase = tf.nn.conv2d(conv4_5_3x3, self.conv4_5_1x1_increase_weights, [1,1,1,1], padding='SAME')
        conv4_5_1x1_increase = self.bn4_5_increase(conv4_5_1x1_increase)
        conv4_5 = tf.nn.relu(tf.add(conv4_5_1x1_increase, conv4_4)) 
      # output shape:[14, 14, 1024]
      with tf.name_scope('conv4_6'): 
        conv4_6_1x1_reduce = tf.nn.conv2d(conv4_5, self.conv4_6_1x1_reduce_weights, [1,1,1,1], padding='SAME')
        conv4_6_1x1_reduce = tf.nn.relu(self.bn4_6_reduce(conv4_6_1x1_reduce))
        conv4_6_3x3 = tf.nn.conv2d(conv4_6_1x1_reduce, self.conv4_6_3x3_weights, [1,1,1,1], padding='SAME')
        conv4_6_3x3 = tf.nn.relu(self.bn4_6_3x3(conv4_6_3x3))
        conv4_6_1x1_increase = tf.nn.conv2d(conv4_6_3x3, self.conv4_6_1x1_increase_weights, [1,1,1,1], padding='SAME')
        conv4_6_1x1_increase = self.bn4_6_increase(conv4_6_1x1_increase)
        conv4_6 = tf.nn.relu(tf.add(conv4_6_1x1_increase, conv4_5)) 
      # output shape:[14, 14, 1024]
      with tf.name_scope('conv5_1'):
        conv5_1_proj = tf.nn.conv2d(conv4_6, self.conv5_1_1x1_proj_weights, [1,2,2,1], padding='SAME')
        conv5_1_proj = self.bn5_1_proj(conv5_1_proj)
        conv5_1_1x1_reduce = tf.nn.conv2d(conv4_6, self.conv5_1_1x1_reduce_weights, [1,2,2,1], padding='SAME')
        conv5_1_1x1_reduce = tf.nn.relu(self.bn5_1_reduce(conv5_1_1x1_reduce))
        conv5_1_3x3 = tf.nn.conv2d(conv5_1_1x1_reduce, self.conv5_1_3x3_weights, [1,1,1,1], padding='SAME')
        conv5_1_3x3 = tf.nn.relu(self.bn5_1_3x3(conv5_1_3x3))
        conv5_1_1x1_increase = tf.nn.conv2d(conv5_1_3x3, self.conv5_1_1x1_increase_weights, [1,1,1,1], padding='SAME')
        conv5_1_1x1_increase = self.bn5_1_increase(conv5_1_1x1_increase)
        conv5_1 = tf.nn.relu(tf.add(conv5_1_1x1_increase, conv5_1_proj))
      # output shape:[7, 7, 2048]
      with tf.name_scope('conv5_2'): 
        conv5_2_1x1_reduce = tf.nn.conv2d(conv5_1, self.conv5_2_1x1_reduce_weights, [1,1,1,1], padding='SAME')
        conv5_2_1x1_reduce = tf.nn.relu(self.bn5_2_reduce(conv5_2_1x1_reduce))
        conv5_2_3x3 = tf.nn.conv2d(conv5_2_1x1_reduce, self.conv5_2_3x3_weights, [1,1,1,1], padding='SAME')
        conv5_2_3x3 = tf.nn.relu(self.bn5_2_3x3(conv5_2_3x3))
        conv5_2_1x1_increase = tf.nn.conv2d(conv5_2_3x3, self.conv5_2_1x1_increase_weights, [1,1,1,1], padding='SAME')
        conv5_2_1x1_increase = self.bn5_2_increase(conv5_2_1x1_increase)
        conv5_2 = tf.nn.relu(tf.add(conv5_2_1x1_increase, conv5_1))
      # output shape:[7, 7, 2048]
      with tf.name_scope('conv5_3'): 
        conv5_3_1x1_reduce = tf.nn.conv2d(conv5_2, self.conv5_3_1x1_reduce_weights, [1,1,1,1], padding='SAME')
        conv5_3_1x1_reduce = tf.nn.relu(self.bn5_3_reduce(conv5_3_1x1_reduce))
        conv5_3_3x3 = tf.nn.conv2d(conv5_3_1x1_reduce, self.conv5_3_3x3_weights, [1,1,1,1], padding='SAME')
        conv5_3_3x3 = tf.nn.relu(self.bn5_3_3x3(conv5_3_3x3))
        conv5_3_1x1_increase = tf.nn.conv2d(conv5_3_3x3, self.conv5_3_1x1_increase_weights, [1,1,1,1], padding='SAME')
        conv5_3_1x1_increase = self.bn5_3_increase(conv5_3_1x1_increase)
        conv5_3 = tf.nn.relu(tf.add(conv5_3_1x1_increase, conv5_2)) 
      # output shape:[7, 7, 2048]
      pool5_7x7_s1 = tf.layers.average_pooling2d(conv5_3, 7, 7, padding='SAME')
      dim = pool5_7x7_s1.get_shape().as_list()[3]
      pool5_7x7_s1 = tf.reshape(pool5_7x7_s1, [-1, dim])
      # output shape: [2048]
      assert pool5_7x7_s1.get_shape().as_list()[1:] == [2048]
    
    return conv3_4, conv4_6, conv5_3, pool5_7x7_s1 # shape of 28,14,7,1

  def avg_pool(self, bottom, name):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

  def max_pool(self, bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

  def conv_layer(self, bottom, name):
    with tf.variable_scope(name):
      filt = self.get_conv_filter(name)
      conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
      conv_biases = self.get_bias(name)
      bias = tf.nn.bias_add(conv, conv_biases)
      relu = tf.nn.relu(bias)
      return relu

  def fc_layer(self, bottom, name):
    with tf.variable_scope(name):
      shape = bottom.get_shape().as_list()
      dim = 1
      for d in shape[1:]:
        dim *= d
      x = tf.reshape(bottom, [-1, dim])
      weights = self.get_fc_weight(name)
      biases = self.get_bias(name)
      # Fully connected layer. Note that the '+' operation automatically
      # broadcasts the biases.
      fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
      return fc

  def get_filter_bias(self, name):
    return tf.Variable(self.data_dict[name]['weights'], name="weights"), \
           tf.constant(self.data_dict[name]['biases'], name="biases")
  
  def get_filter(self, name):
    return tf.Variable(self.data_dict[name]['weights'], name="filter")

  def get_bias(self, name):
    return tf.Variable(self.data_dict[name]['biases'], name="biases")

