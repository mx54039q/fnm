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
                    decay_steps=cfg.decay_steps, decay_rate=0.98,staircase=True) #

                self.train_texture = tf.train.AdamOptimizer(self.lr, beta1=cfg.beta1).minimize(
                    self.texture_loss,
                    global_step=self.global_step, var_list=self.vars_gen)
                self.train_gen = tf.train.AdamOptimizer(self.lr, beta1=cfg.beta1).minimize(
                    self.texture_loss + cfg.lambda_val2*self.g_loss, 
                    global_step=self.global_step, var_list=self.vars_gen)#
                self.train_dis = tf.train.AdamOptimizer(self.lr, beta1=cfg.beta1).minimize(
                    cfg.lambda_val2*self.d_loss,
                    global_step=self.global_step, var_list=self.vars_dis)
                
        tf.logging.info('Seting up the main structure')

    def build_arch(self):
        if(cfg.use_profile):
            with tf.variable_scope('vgg_encoder') as scope:
                self.vgg_pool5, self.vgg_relu7 = self.vgg.forward(self.profile)
        else:
            self.vgg_pool5 = self.profile
        assert self.vgg_pool5.get_shape().as_list()[1:] == [7, 7, 512]
        print 'VGG output feature shape:', self.vgg_pool5.get_shape()
        
        # The feature vector extract from profile by VGG-16 is 4096-D
        with tf.variable_scope('decoder') as scope:
            # Construct BatchNorm Layer
            bn0_2 = batch_norm(name='bn0_2')
            bn1_1 = batch_norm(name='bn1_1')
            bn1_2 = batch_norm(name='bn1_2')
            bn2_1 = batch_norm(name='bn2_1')
            bn2_2 = batch_norm(name='bn2_2')
            bn3_1 = batch_norm(name='bn3_1')
            bn3_2 = batch_norm(name='bn3_2')
            
            # map from fc7_encoder to 14 × 14 × 256 localized features
            #fc1 = fullyConnect(self.fc7_encoder, 14*14*256, 'fc1') # bn0()
            #fc1_reshape = tf.reshape(fc1, [-1,14,14,256])

            # Stacked Transpose Convolutions
            g_input = tf.reshape(self.vgg_relu7, [-1,4,4,256])
            #output shape: [4, 4, 256]
            dconv0_1 = lrelu(deconv2d(g_input, 256, 'dconv0_1', 
                kernel_size=3, strides = 1, padding='valid'))
            print(dconv0_1.get_shape())
            #output shape: [6, 6, 256]
            dconv0_2 = lrelu(bn0_2(deconv2d(dconv0_1, 256, 'dconv0_2', 
                kernel_size=3, strides = 2), self.is_train))
            print(dconv0_2.get_shape())
            #output shape: [12, 12, 256]
            dconv1_1 = lrelu(bn1_1(deconv2d(dconv0_2, 128, 'dconv1_1', 
                kernel_size=3, strides = 1, padding='valid'), self.is_train))
            #output shape: [14, 14, 128]
            dconv1_2 = lrelu(bn1_2(deconv2d(dconv1_1, 128, 'dconv1_2', 
                kernel_size=5, strides = 2), self.is_train))
            #output shape: [28, 28, 128]
            dconv2_1 = lrelu(bn2_1(deconv2d(dconv1_2, 64, 'dconv2_1', 
                kernel_size=5, strides = 2), self.is_train))
            #output shape: [56, 56, 64]
            dconv2_2 = lrelu(bn2_2(deconv2d(dconv2_1, 64, 'dconv2_2', 
                kernel_size=5, strides = 2), self.is_train))
            #output shape: [112, 112, 32]
            dconv3_1 = lrelu(bn3_1(deconv2d(dconv2_2, 32, 'dconv3_1', 
                kernel_size=5, strides = 2), self.is_train))
            #output shape: [224, 224, 32]
            dconv3_2 = lrelu(bn3_2(deconv2d(dconv3_1, 32, 'dconv3_2', 
                kernel_size=5, strides = 1), self.is_train))
            #output shape: [224, 224, 32]
            pw_conv = conv2d(dconv3_2, 3, 'pw_conv', kernel_size=1, strides = 1,
                                  activation = tf.nn.tanh)
            self.texture = (pw_conv + 1) * 127.5
        assert self.texture.get_shape().as_list()[1:] == [224,224,3]
        
        # Map texture and ground truth frontal into features again by VGG    
        with tf.variable_scope('vgg_encoder_recon'):
            self.vgg_pool5_recon, self.vgg_relu7_recon = self.vgg.forward(self.texture)
            self.vgg_pool5_recon_gt, self.vgg_relu7_recon_gt = self.vgg.forward(self.front)
        assert self.vgg_pool5_recon.get_shape().as_list()[1:] == [7,7,512]
        assert self.vgg_relu7_recon.get_shape().as_list()[1] == 4096
        
        # Construct discriminator between generalized front face and ground truth
        self.d_fake, self.d_fake_logits = self.discriminator(self.vgg_pool5_recon, reuse=False)
        self.d_real, self.d_real_logits = self.discriminator(self.vgg_pool5_recon_gt, reuse=True)
        assert self.d_real.get_shape().as_list()[1] == 1
        
    def discriminator(self, fmap, y=None, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse) as scope:
            # input feature maps is 7 x 7 x 512 from vgg-face
            d_bn1 = batch_norm(name='d_bn1')

            h0 = lrelu(conv2d(fmap, 128, 'dis_conv0'))
            h1 = lrelu(d_bn1(conv2d(h0, 64, 'dis_conv1')))
            dim = 1
            for d in h1.get_shape().as_list()[1:]:
                dim *= d
            h2 = fullyConnect(tf.reshape(h1, [-1, dim]), 1, 'dis_fc')

            return tf.nn.sigmoid(h2), h2
            
    def loss(self):
        with tf.name_scope('loss') as scope:
            # 1. Frontalization Loss: L1-Norm
            self.front_loss = tf.reduce_mean(tf.abs(self.front - self.texture)) # 
            tf.add_to_collection('losses', self.front_loss)
            
            # 2. Feature Loss: Cosine-Norm
            vgg_relu7_recon_norm = self.vgg_relu7_recon / tf.norm(self.vgg_relu7_recon,
                axis=1,keep_dims=True)
            vgg_relu7_recon_gt_norm = self.vgg_relu7_recon_gt / tf.norm(self.vgg_relu7_recon_gt,
                axis=1,keep_dims=True)
            self.feature_loss = tf.losses.cosine_distance(labels=vgg_relu7_recon_gt_norm,
                predictions=vgg_relu7_recon_norm, dim=1)
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
