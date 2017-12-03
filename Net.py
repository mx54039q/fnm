#coding: utf-8
"""

"""

import tensorflow as tf

from config import cfg
from utils import loadData
import vgg16
from ops import *

epsilon = 1e-9


class Net(object):
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope('vgg'):
                self.vgg = vgg16.Vgg16()
                self.vgg.build()
            if cfg.is_train:
                if cfg.use_profile:
                    self.profile = tf.placeholder("float", [None, 224, 224, 3], 'profile')
                else:
                    self.profile = tf.placeholder("float", [None, 4096], 'profile')
                self.is_train = tf.placeholder(tf.bool, name='is_train')
                self.front = tf.placeholder("float", [None, 224, 224, 3], 'front')
                #self.Y = tf.one_hot(self.labels, depth=10, axis=1, dtype=tf.float32)
                
                self.build_arch()
                self.loss()
                self._summary()
                
                self.vars_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
                self.vars_dis = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.lr = tf.train.exponential_decay(cfg.lr,self.global_step,
                    decay_steps=cfg.dataset_size/cfg.batch_size*100, decay_rate=0.98,staircase=True) #
                self.optimizer = tf.train.AdamOptimizer(self.lr, epsilon = epsilon)
                self.train_texture = self.optimizer.minimize(self.texture_loss,
                    global_step=self.global_step, var_list=self.vars_gen)
                self.train_gen = self.optimizer.minimize(self.texture_loss + cfg.lambda_val2*self.g_loss,
                    global_step=self.global_step, var_list=self.vars_gen)
                self.train_dis = self.optimizer.minimize(cfg.lambda_val2*self.d_loss,
                    global_step=self.global_step, var_list=self.vars_dis)
                
        tf.logging.info('Seting up the main structure')

    def build_arch(self):
        if(cfg.use_profile):
            with tf.variable_scope('vgg_encoder') as scope:
                self.fc7_encoder = self.vgg.forward(self.profile)
        else:
            self.fc7_encoder = self.profile
        assert self.fc7_encoder.get_shape().as_list()[1] == 4096
        print 'VGG output feature shape:', self.fc7_encoder.get_shape()
        
        # The feature vector extract from profile by VGG-16 is 4096-D
        with tf.variable_scope('decoder') as scope:
            # Construct BatchNorm Layer
            bn0 = batch_norm(name='bn0')
            bn1_1 = batch_norm(name='bn1_1')
            bn1_2 = batch_norm(name='bn1_2')
            bn2_1 = batch_norm(name='bn2_1')
            bn2_2 = batch_norm(name='bn2_2')
            bn3_1 = batch_norm(name='bn3_1')
            bn3_2 = batch_norm(name='bn3_2')
            
            # map from fc7_encoder to 14 × 14 × 256 localized features
            fc1 = fullyConnect(self.fc7_encoder, 14*14*256, 'fc1') # bn0()
            fc1_reshape = tf.reshape(fc1, [-1,14,14,256])
            print 'fc1 output shape:', fc1_reshape.get_shape()

            # Stacked Transpose Convolutions
            dconv1_1 = tf.nn.relu(bn1_1(deconv2d(fc1_reshape, 128, 'dconv1_1', 
                kernel_size=5, strides = 2), self.is_train))
            dconv1_2 = tf.nn.relu(bn1_2(deconv2d(dconv1_1, 128, 'dconv1_2', 
                kernel_size=5, strides = 1), self.is_train))
            dconv2_1 = tf.nn.relu(bn2_1(deconv2d(dconv1_2, 64, 'dconv2_1', 
                kernel_size=5, strides = 2), self.is_train))
            dconv2_2 = tf.nn.relu(bn2_2(deconv2d(dconv2_1, 64, 'dconv2_2', 
                kernel_size=5, strides = 2), self.is_train))
            dconv3_1 = tf.nn.relu(bn3_1(deconv2d(dconv2_2, 32, 'dconv3_1', 
                kernel_size=5, strides = 2), self.is_train))
            dconv3_2 = tf.nn.relu(bn3_2(deconv2d(dconv3_1, 32, 'dconv3_2', 
                kernel_size=5, strides = 1), self.is_train))
            pw_conv = conv2d(dconv3_2, 3, 'pw_conv', kernel_size=1, strides = 1,
                                  activation = tf.nn.tanh)
            self.texture = (pw_conv + 1) * 128.0
        assert self.texture.get_shape().as_list()[1:] == [224,224,3]
        
        # Map texture and ground truth frontal into features again by VGG    
        with tf.variable_scope('vgg_encoder_recon'):
            self.recon_feature = self.vgg.forward(self.texture)
            self.recon_feature_gt = self.vgg.forward(self.front)
        assert self.recon_feature.get_shape().as_list()[1] == 4096
        
        # Construct discriminator between generalized front face and ground truth
        with tf.variable_scope('discriminator') as scope:
            self.d_fake_logits = fullyConnect(self.recon_feature, 1, 'dis')
            self.d_fake = tf.nn.sigmoid(self.d_fake_logits)
            self.d_real_logits = fullyConnect(self.recon_feature_gt, 1, 'dis', reuse=True)
            self.d_real = tf.nn.sigmoid(self.d_real_logits)
        assert self.d_real.get_shape().as_list()[1] == 1
        
    def loss(self):
        with tf.name_scope('loss') as scope:
            # 1. Frontalization Loss: L1-Norm
            self.front_loss = tf.losses.absolute_difference(self.front, self.texture) # 
            tf.add_to_collection('losses', self.front_loss)
            
            # 2. Feature Loss: Cosine-Norm
            recon_feature_norm = self.recon_feature / tf.norm(self.recon_feature,
                                                              axis=1,keep_dims=True)
            recon_feature_gt_norm = self.recon_feature_gt / tf.norm(self.recon_feature_gt,
                                                                    axis=1,keep_dims=True)
            self.feature_loss = tf.losses.cosine_distance(recon_feature_gt_norm,
                                                          recon_feature_norm, dim=1)
            tf.add_to_collection('losses', self.feature_loss)                 
            
            # 3. L2 Regulation Loss
            l2_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='decoder'))
            tf.add_to_collection('losses', l2_loss)
            
            # 4. Adversarial Loss
            d_loss_real = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
                tf.ones_like(self.d_real), logits=self.d_real_logits, label_smoothing=0.1))
            d_loss_fake = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
                tf.zeros_like(self.d_fake), logits=self.d_fake_logits, label_smoothing=0.1))
            self.d_loss = d_loss_real + d_loss_fake
            self.g_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
                tf.ones_like(self.d_fake), logits=self.d_fake_logits, label_smoothing=0.1))
            tf.add_to_collection('losses', self.d_loss)
            tf.add_to_collection('losses', self.g_loss)
            
            # 5. Total texture loss
            self.texture_loss = self.front_loss + cfg.lambda_val1 * self.feature_loss + l2_loss # loss
            

    # Summary
    def _summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/front_loss', self.front_loss))
        train_summary.append(tf.summary.scalar('train/feature_loss', self.feature_loss))
        self.train_summary = tf.summary.merge(train_summary)

        #correct_prediction = tf.equal(tf.to_int32(self.labels), self.argmax_idx)
        #self.batch_accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        #self.test_acc = tf.placeholder_with_default(tf.constant(0.), shape=[])
