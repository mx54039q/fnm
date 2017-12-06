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
            self.data_feed = loadData(batch_size=cfg.batch_size, train_shuffle=True) # False
            with tf.variable_scope('vgg'):
                self.vgg = vgg16.Vgg16()
                self.vgg.build()
            if cfg.is_train:
                #if cfg.use_profile:
                #    self.profile = tf.placeholder("float", [None, 224, 224, 3], 'profile')
                #else:
                #    self.profile = tf.placeholder("float", [None, 4096], 'profile')
                #self.front = tf.placeholder("float", [None, 224, 224, 3], 'front')
                #self.Y = tf.one_hot(self.labels, depth=10, axis=1, dtype=tf.float32)
                
                self.is_train = tf.placeholder(tf.bool, name='is_train')
                self.profile, self.front = self.data_feed.get_train()
                
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
                    self.texture_loss + self.g_loss, 
                    global_step=self.global_step, var_list=self.vars_gen)#
                self.train_dis = tf.train.AdamOptimizer(self.lr, beta1=cfg.beta1).minimize(
                    self.d_loss,
                    global_step=self.global_step, var_list=self.vars_dis)
            else:
                self.profile = tf.placeholder("float", [None, 224, 224, 3], 'profile')
                self.is_train = tf.placeholder(tf.bool, name='is_train')
                self.front = tf.placeholder("float", [None, 224, 224, 3], 'front')
                
                self.build_arch()
                
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
            bn1_1 = batch_norm(name='bn1_1')
            bn1_2 = batch_norm(name='bn1_2')
            bn2_1 = batch_norm(name='bn2_1')
            bn2_2 = batch_norm(name='bn2_2')
            bn3_1 = batch_norm(name='bn3_1')
            #bn3_2 = batch_norm(name='bn3_2')
            
            # map from fc7_encoder to 14 × 14 × 256 localized features
            #fc1 = fullyConnect(self.fc7_encoder, 14*14*256, 'fc1') # bn0()
            #fc1_reshape = tf.reshape(fc1, [-1,14,14,256])

            # Stacked Transpose Convolutions
            fc1 = fullyConnect(self.vgg_relu7, 7*7*256, 'fc1') # bn0()
            g_input = tf.reshape(fc1, [-1,7,7,256])
            #input shape: [7, 7, 256]
            dconv1_1 = lrelu(bn1_1(deconv2d(g_input, 128, 'dconv1_1', 
                kernel_size=5, strides = 2), self.is_train))
            #output shape: [14, 14, 128]
            dconv1_2 = lrelu(bn1_2(deconv2d(dconv1_1, 64, 'dconv1_2', 
                kernel_size=5, strides = 2), self.is_train))
            #output shape: [28, 28, 128]
            dconv2_1 = lrelu(bn2_1(deconv2d(dconv1_2, 64, 'dconv2_1', 
                kernel_size=5, strides = 2), self.is_train))
            #output shape: [56, 56, 64]
            dconv2_2 = lrelu(bn2_2(deconv2d(dconv2_1, 32, 'dconv2_2', 
                kernel_size=5, strides = 2), self.is_train))
            #output shape: [112, 112, 32]
            dconv3_1 = lrelu(bn3_1(deconv2d(dconv2_2, 32, 'dconv3_1', 
                kernel_size=5, strides = 2), self.is_train))
            #output shape: [224, 224, 32]
            pw_conv = conv2d(dconv3_1, 3, 'pw_conv', kernel_size=1, strides = 1,
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
        #real_pf = tf.concat([self.profile, self.front], 3)
        #fake_pf = tf.concat([self.profile, self.texture], 3)
        self.d_real, self.d_real_logits = self.discriminator(self.vgg_pool5_recon, reuse=False)
        self.d_fake, self.d_fake_logits = self.discriminator(self.vgg_pool5_recon_gt, reuse=True)
        assert self.d_real.get_shape().as_list()[1] == 1
        
    def discriminator(self, images, y=None, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse) as scope:
            # shape of input images 224 x 224 x 6, concat profile and front face
            # shape of input images 7 x 7 x 512,
            d_bn1 = batch_norm(name='d_bn1')
            d_bn2 = batch_norm(name='d_bn2')
            
            #images = images / 127.5 - 1
            h0 = lrelu(conv2d(images, 128, 'd_conv0', kernel_size=3))
            # h0 is (7 x 7 x 128)
            h1 = lrelu(d_bn1(conv2d(h0, 128, 'd_conv1', kernel_size=3), self.is_train))
            # h1 is (7 x 7 x 128)
            h2 = lrelu(d_bn2(conv2d(h1, 256, 'd_conv2', kernel_size=3), self.is_train))
            # h2 is (7 x 7 x 256)
            dim = 1
            for d in h2.get_shape().as_list()[1:]:
                dim *= d
            h3 = fullyConnect(tf.reshape(h2, [-1, dim]), 1, 'd_fc1')

            return tf.nn.sigmoid(h3), h3

    def loss(self):
        with tf.name_scope('loss') as scope:
            # 1. Frontalization Loss: L1-Norm
            self.front_loss = tf.reduce_mean(tf.abs(self.front/255. - self.texture/255.)) # 
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
            self.texture_loss = cfg.lambda_l1 * self.front_loss + cfg.lambda_fea * self.feature_loss + l2_loss
            

    # Summary
    def _summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/front_loss', self.front_loss))
        train_summary.append(tf.summary.scalar('train/feature_loss', self.feature_loss))
        self.train_summary = tf.summary.merge(train_summary)

        #correct_prediction = tf.equal(tf.to_int32(self.labels), self.argmax_idx)
        #self.batch_accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        #self.test_acc = tf.placeholder_with_default(tf.constant(0.), shape=[])
