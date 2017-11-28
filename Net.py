#coding: utf-8
"""

"""

import tensorflow as tf

from config import cfg
from utils import loadData
#from capsLayer import CapsLayer
import vgg16
from ops import *

epsilon = 1e-9


class Net(object):
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope('vgg'):
                self.vgg = vgg16.Vgg16()
                self.vgg.build()
            if is_training:
                if use_profile:
                    self.profile = tf.placeholder("float", [None, 224, 224, 3])
                else:
                    self.profile = tf.placeholder("float", [None, 4096])
                self.front = tf.placeholder("float", [None, 224, 224, 3])
                #self.Y = tf.one_hot(self.labels, depth=10, axis=1, dtype=tf.float32)
                
                self.build_arch()
                self.loss()
                self._summary()
                
                self.t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.lr = tf.train.exponential_decay(cfg.lr,self.global_step,
                    decay_steps=100,decay_rate=0.98,staircase=True)
                self.optimizer = tf.train.AdamOptimizer(self.lr)
                self.train_op = self.optimizer.minimize(self.total_loss,
                                  global_step=self.global_step, var_list=self.t_vars)

        tf.logging.info('Seting up the main structure')

    def build_arch(self):
        if(self.use_profile):
            with tf.name_scope('vgg_encoder') as scope:
                self.fc7_encoder = self.vgg.forward(self.profile)
        else:
            self.fc7_encoder = self.profile
            
        # The feature vector extract from profile by VGG-16 is 4096-D
        assert self.fc7_encoder.get_shape().as_list()[1] == 4096
        print 'VGG output feature shape:', self.fc7_encoder.get_shape()
        with tf.variable_scope('decoder') as scope:
            # Construct BatchNorm Layer
            bn1_1 = batch_norm(name='bn1_1')
            bn1_2 = batch_norm(name='bn1_2')
            bn2_1 = batch_norm(name='bn2_1')
            bn2_2 = batch_norm(name='bn2_2')
            bn3_1 = batch_norm(name='bn3_1')
            bn3_2 = batch_norm(name='bn3_2')
            
            # map from fc7_encoder to 14 × 14 × 256 localized features
            fc1 = fullyConnect(self.fc7_encoder, 14*14*256, 'fc1')
            fc1_reshape = tf.reshape(fc1, [-1,14,14,256])
            print 'fc1 output shape:', fc1_reshape.get_shape()

            # Stacked Transpose Convolutions
            dconv1_1 = bn1_1(deconv2d(fc1_reshape, filters=128, kernel_size=5, strides = 2,
                                  activation = tf.nn.relu, name = 'dconv1_1'))
            dconv1_2 = bn1_2(deconv2d(dconv1_1, filters=128, kernel_size=5, strides = 1,
                                  activation = tf.nn.relu, name = 'dconv1_2'))
            dconv2_1 = bn2_1(deconv2d(dconv1_2, filters=64, kernel_size=5, strides = 2,
                                  activation = tf.nn.relu, name = 'dconv2_1'))
            dconv2_2 = bn2_2(deconv2d(dconv2_1, filters=64, kernel_size=5, strides = 2,
                                  activation = tf.nn.relu, name = 'dconv2_2'))
            dconv3_1 = bn3_1(deconv2d(dconv2_2, filters=32, kernel_size=5, strides = 2,
                                  activation = tf.nn.relu, name = 'dconv3_1'))
            dconv3_2 = bn3_2(deconv2d(dconv3_1, filters=32, kernel_size=5, strides = 1,
                                  activation = tf.nn.relu, name = 'dconv3_2'))
            pw_conv = conv2d(dconv3_2, filters=3, kernel_size=1, strides = 1,
                                  activation = tf.nn.tanh, name='pw_conv')
            self.texture = (pw_conv + 1) * 127.5
        assert self.texture.get_shape().as_list()[1:] == [224,224,3]
        
        # Map texture and ground truth frontal into features again by VGG    
        with tf.variable_scope('vgg_encoder_recon'):
            self.recon_feature = self.vgg.forward(self.texture)
            self.recon_feature_gt = self.vgg.forward(self.front)
        assert self.recon_feature.get_shape().as_list()[1] == 4096
            
    def loss(self):
        # 1. Frontalization Loss: L1-Norm
        self.front_loss = tf.losses.absolute_difference(self.front, self.texture)
        
        # 2. Feature Loss: Cosine-Norm
        recon_feature_norm = self.recon_feature / tf.norm(self.recon_feature,
                                                          axis=1,keep_dims=True)
        recon_feature_gt_norm = self.recon_feature_gt / tf.norm(self.recon_feature_gt,
                                                                axis=1,keep_dims=True)
        self.feature_loss = tf.losses.cosine_distance(recon_feature_gt_norm,
                                                      recon_feature_norm, dim=1)
        
        # 3. Total loss
        self.total_loss = self.front_loss + cfg.lambda_val * self.feature_loss

    # Summary
    def _summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/front_loss', self.front_loss))
        train_summary.append(tf.summary.scalar('train/feature_loss', self.feature_loss))
        train_summary.append(tf.summary.scalar('train/total_loss', self.total_loss))
        recon_img = self.texture
        train_summary.append(tf.summary.image('front_img', recon_img))
        self.train_summary = tf.summary.merge(train_summary)

        #correct_prediction = tf.equal(tf.to_int32(self.labels), self.argmax_idx)
        #self.batch_accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        #self.test_acc = tf.placeholder_with_default(tf.constant(0.), shape=[])
