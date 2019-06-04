# coding: utf-8
# --------------------------------------------------------
# FNM
# Written by Yichen Qian
# --------------------------------------------------------

import tensorflow as tf
from PIL import Image
from config import cfg
from utils import loadData
from resnet50 import Resnet50
from ops import *
import tensorflow.contrib.slim as slim

epsilon = 1e-9

class WGAN_GP(object):
  """Class for Face Normalization Model
  
  This class is for face normalization task, including the following three contributions.
  
  1. Feature-embedded: Embedding pretrained face recognition model in G.
  2. Attention Mechanism: Attention Discriminator for elaborate image.
  3. Pixel Loss: front-to-front transform introduce pixel-wise loss.

  """
  def __init__(self):
    self.graph = tf.get_default_graph()
    #with self.graph.as_default():
    self.batch_size = cfg.batch_size
    self.is_train = tf.placeholder(tf.bool, name='is_train')

  def build_up(self, profile, front):
    """Build up architecture
    
    1. Pretrained face recognition model forward
    2. Decoder from feature to image
    3. Refeed generated image to face recognition model
    4. Feed generated image to Discriminator
    5. Construct 'Grade Penalty' for discriminator
    
    """
    
    #with self.graph.as_default():
    # Construct Template Model (G_enc) to encoder input face
    with tf.variable_scope('face_model'):
      self.face_model = Resnet50()
      self.face_model.build()
      print('VGG model built successfully.')
    
    # Use pretrained model(vgg-face) as encoder of Generator
    self.feature_p = self.face_model.forward(profile,'profile_enc')
    self.feature_f = self.face_model.forward(front, 'front_enc')
    print 'Face model output feature shape:', self.feature_p[-1].get_shape()
    
    # Decoder front face from vgg feature
    self.gen_p = self.decoder(self.feature_p)
    self.gen_f = self.decoder(self.feature_f, reuse=True)
    print 'Generator output shape:', self.gen_p.get_shape()
    
    # Map texture into features again by VGG  
    self.feature_gen_p = self.face_model.forward(self.gen_p,'profile_gen_enc')
    self.feature_gen_f = self.face_model.forward(self.gen_f, 'front_gen_enc')
    print 'Feature of Generated Image shape:', self.feature_gen_p[-1].get_shape()
    
    # Construct discriminator between generalized front face and ground truth
    self.dr = self.discriminator(front)
    self.df1 = self.discriminator(self.gen_p, reuse=True)
    self.df2 = self.discriminator(self.gen_f, reuse=True)
    
    # Gradient Penalty #
    with tf.name_scope('gp'):
      alpha = tf.random_uniform((self.gen_p.get_shape().as_list()[0], 1, 1, 1),minval = 0., maxval = 1.,)
      inter = front + alpha * (self.gen_p - front)
      d = self.discriminator(inter, reuse=True)
      grad = tf.gradients([d], [inter])[0]
      slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), [1,2,3]))
      self.gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.))
      ######
      self.grad4 = tf.reduce_mean(slopes)
    
    # Get Viraibles
    all_vars = tf.trainable_variables()
    self.vars_gen = [var for var in all_vars if var.name.startswith('decoder')]
    self.vars_dis = [var for var in all_vars if var.name.startswith('discriminator')]
    self.loss(profile, front)
             
    # Ops for debug
    with tf.name_scope('Debug'):
      grad1 = tf.gradients([self.feature_loss], [self.gen_p])[0]
      self.grad1 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(grad1), [1,2,3])))
      grad2 = tf.gradients([self.g_loss], [self.gen_p])[0]
      self.grad2 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(grad2), [1,2,3])))
      grad3 = tf.gradients([self.front_loss], [self.gen_f])[0]
      self.grad3 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(grad3), [1,2,3])))
      
    # Summary
    self._summary()  
    
    # Optimizer
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.train_gen = tf.train.AdamOptimizer(cfg.lr, beta1=cfg.beta1, beta2=cfg.beta2).minimize(
             self.gen_loss,
             global_step=self.global_step, var_list=self.vars_gen)
    self.train_dis = tf.train.AdamOptimizer(cfg.lr, beta1=cfg.beta1, beta2=cfg.beta2).minimize(
             self.dis_loss,
             global_step=self.global_step, var_list=self.vars_dis)
        
  def decoder(self, feature, reuse=False):
    """Decoder part of generator
    
    Embed pretrained face recognition model in Generator.
    This part transforms feature to image. 
    
    args: 
      feature: face identity feature from pretrained face model.
      reuse: Whether to reuse the model(Default False).
    return: 
      generated front face, which value is in range [0, 255].
      
    """
    
    # The feature vector extracted from profile by Resnet-50 is 2048-D
    with tf.variable_scope('decoder', reuse=reuse) as scope:
      # Choose Normalization Method
      norm = bn if(cfg.norm=='bn') else pixel_norm
      
      # Split feature tuple
      feat28,feat14,feat7,pool5 = feature[0],feature[1],feature[2],feature[3]
      
      # We use the last second feature layer of face model, whose shape is
      # 7 x 7 x 2048. If you use 'flatten feature', you should connect 
      # 'Fully Connect Layer' first to expand dimension of feature. You can
      # use code below and modify the whole network.
      
      # input shape: [7, 7, 2048]
      with tf.variable_scope('conv1'):
        feat7 = tf.nn.relu(norm(conv2d(feat7, 512, 'conv1', 
                                kernel_size=1, strides = 1),self.is_train,'norm1'))
      # ouput shape: [7, 7, 512]
      res1_1 = res_block(feat7, 'res1_1',self.is_train, cfg.norm)
      res1_2 = res_block(res1_1, 'res1_2',self.is_train, cfg.norm)
      res1_3 = res_block(res1_2, 'res1_3',self.is_train, cfg.norm)
      res1_4 = res_block(res1_3, 'res1_4',self.is_train, cfg.norm)
      #ouput shape: [7, 7, 512]
      with tf.variable_scope('dconv2'):
        dconv2 = tf.nn.relu(norm(deconv2d(res1_4, 256, 'dconv2', 
                            kernel_size=4, strides = 2),self.is_train,'norm2'))
      res2 = res_block(dconv2, 'res2',self.is_train, cfg.norm)
      #ouput shape: [14, 14, 256]
      with tf.variable_scope('dconv3'):
        dconv3 = tf.nn.relu(norm(deconv2d(res2, 128, 'dconv3', 
                            kernel_size=4, strides = 2),self.is_train,'norm3'))
      res3 = res_block(dconv3, 'res3',self.is_train, cfg.norm)
      #output shape: [28, 28, 128]
      with tf.variable_scope('dconv4'):
        dconv4 = tf.nn.relu(norm(deconv2d(res3, 64, 'dconv4', 
                            kernel_size=4, strides = 2),self.is_train,'norm4'))
      res4 = res_block(dconv4, 'res4',self.is_train, cfg.norm)
      #output shape: [56, 56, 64]
      with tf.variable_scope('dconv5'):
        dconv5 = tf.nn.relu(norm(deconv2d(res4, 32, 'dconv5',
                            kernel_size=4, strides = 2),self.is_train,'norm5'))
      res5 = res_block(dconv5, 'res5',self.is_train, cfg.norm)
      #input shape: [112, 112, 32]
      with tf.variable_scope('dconv6'):
        dconv6 = tf.nn.relu(norm(deconv2d(res5, 32, 'dconv6', 
                            kernel_size=4, strides = 2),self.is_train,'norm6'))
      res6 = res_block(dconv6, 'res6',self.is_train, cfg.norm)
      #output shape: [224, 224, 32]
      with tf.variable_scope('cw_conv'):
        gen = tf.nn.tanh(conv2d(res6, 3, 'pw_conv', kernel_size=1, strides = 1))
    
      return (gen + 1) * 127.5
    
  def discriminator(self, images, reuse=False):
    """Attention Discriminator Networks
    
    As normalized face is strictly aligned, we construct 
    Attention Discriminators on fixed area of face image 
    (i.e. whole image, eyes, nose, mouth, face). This 
    contribution make Discriminator focus on local area, 
    and make Generator produce more elaborate detail.     
    
    1. Waasertein Distance
    2. Layer Normalization, LReLU, Stride Convolution
    
    args: 
      image: front face in range [0,255].
      reuse: Whether to reuse the model(Default False).
    return: 
      a set of and logits.
    """
    
    with tf.variable_scope("discriminator", reuse=reuse) as scope:
      norm = slim.layer_norm
      
      images = images / 127.5 - 1
      bs = images.get_shape().as_list()[0]
      
      ###################### Note ########################
      # Four Fixed Area. Modify them to fit your dataset #
      ####################################################
      
      # eyes = tf.slice(images, [0,64,50,0], [bs,36,124,cfg.channel])
      # nose = tf.slice(images, [0,75,90,0], [bs,65,44,cfg.channel])
      # mouth = tf.slice(images, [0,140,75,0], [bs,30,74,cfg.channel])
      # face = tf.slice(images, [0,64,50,0], [bs,116,124,cfg.channel])
      
      # modified
      eyes = tf.slice(images, [0,56,44,0], [bs,46,136,cfg.channel])  # [56:102,44:180,:]
      nose = tf.slice(images, [0,70,88,0], [bs,74,48,cfg.channel])  # [70:144,88:136,:]
      mouth = tf.slice(images, [0,144,80,0], [bs,36,64,cfg.channel])  # [144:180,80:144,:]
      face = tf.slice(images, [0,40,34,0], [bs,150,156,cfg.channel])  # [40:190,34:190,:]
      
      with tf.variable_scope("images"):
        with tf.variable_scope('d_conv0'):
          h0_0 = lrelu(conv2d(images, 32, 'd_conv0', kernel_size=3, strides=2))
        # h0 is (112 x 112 x 32)
        with tf.variable_scope('d_conv1'):
          h0_1 = lrelu(norm(conv2d(h0_0, 64, 'd_conv1', kernel_size=3, strides=2)))
        # h1 is (56 x 56 x 64)
        with tf.variable_scope('d_conv2'):
          h0_2 = lrelu(norm(conv2d(h0_1, 128, 'd_conv2', kernel_size=3, strides=2)))
        # h2 is (28 x 28 x 128)
        with tf.variable_scope('d_conv3'):
          h0_3 = lrelu(norm(conv2d(h0_2, 256, 'd_conv3', kernel_size=3, strides=2)))
        # h3 is (14 x 14 x 256)
        with tf.variable_scope('d_conv4'):
          h0_4 = lrelu(norm(conv2d(h0_3, 256, 'd_conv4', kernel_size=3, strides=2)))
        # h4 is (7 x 7 x 256)
        with tf.variable_scope('d_fc'):
          h0_4 = tf.reshape(h0_4, [bs, -1])
          h0_5 = fullyConnect(h0_4, 1, 'd_fc')
        # h5 is (1)
        
      with tf.variable_scope("eyes"):
        with tf.variable_scope('d_conv0'):
          h1_0 = lrelu(conv2d(eyes, 32, 'd_conv0', kernel_size=3, strides=2))
        # h0 is (18 x 62 x 32)
        with tf.variable_scope('d_conv1'):
          h1_1 = lrelu(norm(conv2d(h1_0, 64, 'd_conv1', kernel_size=3, strides=2)))
        # h1 is (9 x 31 x 64)
        with tf.variable_scope('d_conv2'):
          h1_2 = lrelu(norm(conv2d(h1_1, 128, 'd_conv2', kernel_size=3, strides=2)))
        # h2 is (5 x 15 x 128)
        with tf.variable_scope('d_conv3'):
          h1_3 = lrelu(norm(conv2d(h1_2, 256, 'd_conv3', kernel_size=3, strides=2)))
        # h3 is (3 x 8 x 256)
        with tf.variable_scope('d_fc'):
          h1_3 = tf.reshape(h1_3, [bs, -1])
          h1_4 = fullyConnect(h1_3, 1, 'd_fc')
        # h4 is (1)
        
      with tf.variable_scope("nose"):
        with tf.variable_scope('d_conv0'):
          h2_0 = lrelu(conv2d(nose, 32, 'd_conv0', kernel_size=3, strides=2))
        # h0 is (33 x 22 x 32)
        with tf.variable_scope('d_conv1'):
          h2_1 = lrelu(norm(conv2d(h2_0, 64, 'd_conv1', kernel_size=3, strides=2)))
        # h1 is (17 x 11 x 64)
        with tf.variable_scope('d_conv2'):
          h2_2 = lrelu(norm(conv2d(h2_1, 128, 'd_conv2', kernel_size=3, strides=2)))
        # h2 is (9 x 6 x 128)
        with tf.variable_scope('d_conv3'):
          h2_3 = lrelu(norm(conv2d(h2_2, 256, 'd_conv3', kernel_size=3, strides=2)))
        # h3 is (5 x 3 x 256)
        with tf.variable_scope('d_fc'):
          h2_3 = tf.reshape(h2_3, [bs, -1])
          h2_4 = fullyConnect(h2_3, 1, 'd_fc')
        # h4 is (1)
        
      with tf.variable_scope("mouth"):
        with tf.variable_scope('d_conv0'):
          h3_0 = lrelu(conv2d(mouth, 32, 'd_conv0', kernel_size=3, strides=2))
        # h0 is (15 x 37 x 32)
        with tf.variable_scope('d_conv1'):
          h3_1 = lrelu(norm(conv2d(h3_0, 64, 'd_conv1', kernel_size=3, strides=2)))
        # h1 is (8 x 19 x 64)
        with tf.variable_scope('d_conv2'):
          h3_2 = lrelu(norm(conv2d(h3_1, 128, 'd_conv2', kernel_size=3, strides=2)))
        # h2 is (4 x 10 x 128)
        with tf.variable_scope('d_conv3'):
          h3_3 = lrelu(norm(conv2d(h3_2, 256, 'd_conv3', kernel_size=3, strides=2)))
        # h3 is (2 x 5 x 256)
        with tf.variable_scope('d_fc'):
          h3_3 = tf.reshape(h3_3, [bs, -1])
          h3_4 = fullyConnect(h3_3, 1, 'd_fc')
        # h4 is (1)
        
      with tf.variable_scope("face"):
        with tf.variable_scope('d_conv0'):
          h4_0 = lrelu(conv2d(face, 32, 'd_conv0', kernel_size=3, strides=2))
        # h0 is (58 x 62 x 32)
        with tf.variable_scope('d_conv1'):
          h4_1 = lrelu(norm(conv2d(h4_0, 64, 'd_conv1', kernel_size=3, strides=2)))
        # h1 is (29 x 31 x 64)
        with tf.variable_scope('d_conv2'):
          h4_2 = lrelu(norm(conv2d(h4_1, 128, 'd_conv2', kernel_size=3, strides=2)))
        # h2 is (15 x 16 x 128)
        with tf.variable_scope('d_conv3'):
          h4_3 = lrelu(norm(conv2d(h4_2, 256, 'd_conv3', kernel_size=3, strides=2)))
        # h3 is (8 x 8 x 256)
        with tf.variable_scope('d_fc'):
          h4_3 = tf.reshape(h4_3, [bs, -1])
          h4_4 = fullyConnect(h4_3, 1, 'd_fc')
        # h4 is (1)
      
      return h0_5, h1_4, h2_4, h3_4, h4_4

  def loss(self, profile, front):
    """Loss Functions
    
    1. Pixel-Wise Loss: front-to-front reconstruct
    2. Perceptual Loss: Feature distance on space of pretrined face model
    3. Regulation Loss: L2 weight regulation
    4. Adversarial Loss: Wasserstein Distance
    5. Symmetric Loss: NOT APPLY
    6. Drift Loss: NOT APPLY
    7. Grade Penalty Loss: Grade penalty for Discriminator
    """
    
    with tf.name_scope('loss') as scope:
      with tf.name_scope('Feature_Norm'):
        pool5_p_norm = self.feature_p[-1] / (tf.norm(self.feature_p[-1], axis=1,keep_dims=True) + epsilon)
        pool5_f_norm = self.feature_f[-1] / (tf.norm(self.feature_f[-1], axis=1,keep_dims=True) + epsilon)
        pool5_gen_p_norm = self.feature_gen_p[-1] / (tf.norm(self.feature_gen_p[-1], axis=1,keep_dims=True) + epsilon)
        pool5_gen_f_norm = self.feature_gen_f[-1] / (tf.norm(self.feature_gen_f[-1], axis=1,keep_dims=True) + epsilon)
            
      # 1. Frontalization Loss: L1-Norm
      self.front_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(front/255. - self.gen_f/255.), [1,2,3]))
      
      # 2. Feature Loss: Cosine-Norm / L2-Norm
      with tf.name_scope('Perceptual_Loss'):
        self.feature_distance = 0.5*(1 - tf.reduce_sum(tf.multiply(pool5_p_norm, pool5_gen_p_norm), [1])) + \
                                0.5*(1 - tf.reduce_sum(tf.multiply(pool5_f_norm, pool5_gen_f_norm), [1]))
        self.feature_loss = tf.reduce_mean(self.feature_distance)
        tf.add_to_collection('losses', self.feature_loss)
      #with tf.name_scope('Perceptual_Loss'):
      #  self.feature_distance = tf.reduce_mean(tf.square(self.feature_p[-1] - self.feature_gen_p[-1]), [1]) + \
      #                          tf.reduce_mean(tf.square(self.feature_f[-1] - self.feature_gen_f[-1]), [1])
      #  self.feature_loss = tf.reduce_mean(self.feature_distance) #/ 2 
      #  tf.add_to_collection('losses', self.feature_loss)
      
      # 3. L2 Regulation Loss
      with tf.name_scope('Regularation_Loss'):
        self.reg_gen = tf.contrib.layers.apply_regularization(
          tf.contrib.layers.l2_regularizer(cfg.lambda_reg),
          weights_list=[var for var in self.vars_gen if 'kernel' in var.name]
        )
        tf.add_to_collection('losses', self.reg_gen)
        self.reg_dis = tf.contrib.layers.apply_regularization(
          tf.contrib.layers.l2_regularizer(cfg.lambda_reg),
          weights_list=[var for var in self.vars_dis if 'kernel' in var.name]
        )
        tf.add_to_collection('losses', self.reg_dis)
      
      # 4. Adversarial Loss
      with tf.name_scope('Adversarial_Loss'):
        self.d_loss = tf.reduce_mean(tf.add_n(self.df1)*0.5 + tf.add_n(self.df2)*0.5 - tf.add_n(self.dr)) / 5
        self.g_loss = - tf.reduce_mean(tf.add_n(self.df1)*0.5 + tf.add_n(self.df2)*0.5) / 5
        tf.add_to_collection('losses', self.d_loss)
        tf.add_to_collection('losses', self.g_loss)
      
      # 5. Symmetric Loss - not applied
      with tf.name_scope('Symmetric_Loss'):
        mirror_p = tf.reverse(self.gen_p, axis=[2])
        self.sym_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(mirror_p/225. - self.gen_p/255.), [1,2,3]))
      
      # 6. Drift Loss - not applied
      with tf.name_scope('Drift_Loss'):
        self.drift_loss = 0
        #tf.reduce_mean(tf.add_n(tf.square(self.df)) + tf.add_n(tf.square(self.dr))) / 10
      
      # 7. Total Loss
      with tf.name_scope('Total_Loss'):  #
        self.gen_loss = cfg.lambda_l1 * self.front_loss + cfg.lambda_fea * self.feature_loss + \
                cfg.lambda_gan * self.g_loss + self.reg_gen
        self.dis_loss = cfg.lambda_gan * self.d_loss + cfg.lambda_gp * self.gradient_penalty + \
                self.reg_dis
        
  def _summary(self):
    """Tensorflow Summary"""
    
    train_summary = []
    train_summary.append(tf.summary.scalar('train/d_loss', self.d_loss))
    train_summary.append(tf.summary.scalar('train/g_loss', self.g_loss))
    train_summary.append(tf.summary.scalar('train/gp', self.grad4))
    train_summary.append(tf.summary.scalar('train/feature_loss', self.feature_loss))
    train_summary.append(tf.summary.scalar('train/grad_feature', self.grad1))
    train_summary.append(tf.summary.scalar('train/grad_D', self.grad2))
    self.train_summary = tf.summary.merge(train_summary)
    
    
if '__name__' == '__main__':
  net = WGAN_GP()
