#coding: utf-8
"""

"""

import tensorflow as tf
from PIL import Image
from config import cfg
from utils import loadData
from vgg16 import Vgg16
from resnet50 import Resnet50
from ops import *

epsilon = 1e-9


class Net(object):
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.data_feed = loadData(batch_size=cfg.batch_size, train_shuffle=True) # False
            with tf.variable_scope('face_model'):
                self.face_model = Resnet50() # Vgg16()
                self.face_model.build()
            if cfg.is_train:
                #self.front = tf.placeholder("float", [None, 224, 224, 3], 'front')
                #self.Y = tf.one_hot(self.labels, depth=10, axis=1, dtype=tf.float32)
                
                self.is_train = tf.placeholder(tf.bool, name='is_train')
                self.profile, self.front, self.resized_56, self.resized_112 = self.data_feed.get_train() #
                
                self.build_arch()
                self.loss()
                self._summary()
                
                #self.vars_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
                #self.vars_dis = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
                all_vars = tf.trainable_variables()
                self.vars_gen = [var for var in all_vars if var.name.startswith('decoder')]
                self.vars_dis = [var for var in all_vars if var.name.startswith('discriminator')]
                
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.lr = tf.train.exponential_decay(cfg.lr,self.global_step,
                    decay_steps=cfg.decay_steps, decay_rate=0.98,staircase=True) #

                self.train_texture = tf.train.AdamOptimizer(cfg.lr, beta1=cfg.beta1).minimize(
                    self.texture_loss, 
                    global_step=self.global_step, var_list=self.vars_gen)
                self.train_gen = tf.train.AdamOptimizer(cfg.lr, beta1=cfg.beta1).minimize(
                    self.gen_loss,
                    global_step=self.global_step, var_list=self.vars_gen)#
                self.train_dis = tf.train.AdamOptimizer(cfg.lr, beta1=cfg.beta1).minimize(
                    self.dis_loss,
                    global_step=self.global_step, var_list=self.vars_dis)
            else:
                self.profile = tf.placeholder("float", [None, 224, 224, 3], 'profile')
                self.is_train = tf.placeholder(tf.bool, name='is_train')
                self.front = tf.placeholder("float", [None, 224, 224, 3], 'front')
                
                self.build_arch()
                
        tf.logging.info('Seting up the main structure')

    def build_arch(self):
        # Use pretrained model(vgg-face) as encoder of Generator
        with tf.name_scope('face_encoder') as scope:
            _, self.enc_fea = self.face_model.forward(self.profile)
        #assert self.enc_fea.get_shape().as_list()[1:] == [2048]
        print 'Face model output feature shape:', self.enc_fea.get_shape()
        
        # Decoder front face from vgg feature
        self.texture_56, self.texture_112, self.texture_224 = self.decoder(self.enc_fea)
        assert self.texture_224.get_shape().as_list()[1:] == [224,224,3]
        
        # Map texture and ground truth frontal into features again by VGG    
        with tf.name_scope('encoder_recon'):
            print "VGG perceptual removed"
            _, self.enc_fea_recon = self.face_model.forward(self.texture_224, reuse=True)
            #_, self.enc_fea_recon_gt = self.face_model.forward(self.front, reuse=True)
        assert self.enc_fea_recon.get_shape().as_list()[1:] == [2048]
        #assert self.enc_relu7_recon.get_shape().as_list()[1] == 
        
        # Construct discriminator between generalized front face and ground truth
        #real_pf = tf.concat([self.profile, self.front], 3)
        #fake_pf = tf.concat([self.profile, self.texture_224], 3)
        self.dr_56, self.dr_56_logits,self.dr_112, self.dr_112_logits,self.dr_224, self.dr_224_logits = \
            self.discriminator(self.resized_56,self.resized_112,self.front) #
        self.df_56, self.df_56_logits,self.df_112, self.df_112_logits,self.df_224, self.df_224_logits = \
            self.discriminator(self.texture_56,self.texture_112,self.texture_224, reuse=True)
        #assert self.d_real.get_shape().as_list()[1] == 1
    
    def decoder(self, feature, y=None, reuse=False):
        '''
        decoder of generator, decoder feature from vgg
        input: face identity feature from VGG-16 / VGG-res50
        return: generated front face [0, 255].
        '''
        # The feature vector extract from profile by VGG-16 is 4096-D
        # The feature vector extract from profile by Resnet-50 is 2048-D
        with tf.variable_scope('decoder', reuse=reuse) as scope:
            # Construct BatchNorm Layer
            #bn0 = batch_norm(name='bn0')
            bn0 = batch_norm(name='bn0')
            bn1 = batch_norm(name='bn1')
            bn2 = batch_norm(name='bn2')
            bn3 = batch_norm(name='bn3')
            bn4 = batch_norm(name='bn4')
            bn5 = batch_norm(name='bn5')
            bn6 = batch_norm(name='bn6')
            bn7 = batch_norm(name='bn7')
            res1_bn1 = batch_norm(name='res1_bn1')
            res1_bn2 = batch_norm(name='res1_bn2')
            res1_bn3 = batch_norm(name='res1_bn3')
            res2_bn1 = batch_norm(name='res2_bn1')
            res2_bn2 = batch_norm(name='res2_bn2')
            res2_bn3 = batch_norm(name='res2_bn3')
            
            # map from fc7_encoder to 7 × 7 × 256 localized features
            #fc1 = fullyConnect(feature, 7 * 7 * 256, 'fc1') # bn0()
            #fc1 = tf.reshape(fc1, [-1,7,7,256])

            # Stacked Transpose Convolutions:(2048)
            g_input = tf.reshape(feature, [-1, 1, 1, 2048])
            with tf.variable_scope('dconv0'):
                dconv0 = tf.nn.relu(deconv2d(g_input, 512, 'dconv0', 
                                    kernel_size=4, strides = 1, padding='valid'))
            #ouput shape: [4, 4, 512]
            with tf.variable_scope('dconv1'):
                dconv1 = tf.nn.relu(bn1(deconv2d(dconv0, 512, 'dconv1', 
                                        kernel_size=4, strides = 1, padding='valid'), self.is_train))
            #input shape: [7, 7, 512]
            with tf.variable_scope('dconv3'):
                dconv3 = tf.nn.relu(bn3(deconv2d(dconv1, 256, 'dconv3', 
                                        kernel_size=4, strides = 2), self.is_train))
            #ouput shape: [14, 14, 256]
            with tf.variable_scope('dconv4'):
                dconv4 = tf.nn.relu(bn4(deconv2d(dconv3, 128, 'dconv4', 
                                        kernel_size=4, strides = 2), self.is_train))
            #output shape: [28, 28, 128]
            with tf.variable_scope('dconv5'):
                dconv5 = tf.nn.relu(bn5(deconv2d(dconv4, 64, 'dconv5', 
                                        kernel_size=4, strides = 2), self.is_train))
            #output shape: [56, 56, 64]
            with tf.variable_scope('dconv5_concat'):
                fc1 = fullyConnect(feature, 56 * 56, 'fc1') # bn0()
                fc1 = tf.reshape(fc1, [-1,56,56,1])
                fc1 = tf.tile(fc1, [1, 1, 1, 16])
                dconv5_concat = tf.concat([dconv5, fc1], 3)
            #output shape: [56, 56, 80]
            with tf.variable_scope('res1'):
                res1_conv1 = tf.nn.relu(res1_bn1(conv2d(dconv5_concat, 128, 'conv1', 
                                        kernel_size=4, strides = 1), self.is_train))
                res1_conv2 = res1_bn2(conv2d(res1_conv1, 80, 'conv2', 
                                        kernel_size=4, strides = 1), self.is_train)
                res1 = tf.nn.relu(tf.add(dconv5_concat, res1_conv2))
            #output shape: [56, 56, 80]
            with tf.variable_scope('dconv6'):
                dconv6 = tf.nn.relu(bn6(deconv2d(res1, 32, 'dconv6', 
                                        kernel_size=4, strides = 2), self.is_train))
            #output shape: [112, 112, 32]
            with tf.variable_scope('dconv6_concat'):
                fc2 = fullyConnect(feature, 112 * 112, 'fc2')
                fc2 = tf.reshape(fc2, [-1,112,112,1])
                fc2 = tf.tile(fc2, [1, 1, 1, 8])
                dconv6_concat = tf.concat([dconv6, fc2], 3)
            #output shape: [112, 112, 40]
            with tf.variable_scope('res2'):
                res2_conv1 = tf.nn.relu(res2_bn1(conv2d(dconv6_concat, 64, 'conv1', 
                                        kernel_size=4, strides = 1), self.is_train))
                res2_conv2 = res2_bn2(conv2d(res2_conv1, 40, 'conv2', 
                                        kernel_size=4, strides = 1), self.is_train)
                res2 = tf.nn.relu(tf.add(dconv6_concat, res2_conv2))
            #input shape: [112, 112, 40]
            with tf.variable_scope('dconv7'):
                dconv7 = tf.nn.relu(bn7(deconv2d(res2, 32, 'dconv7', 
                                        kernel_size=4, strides = 2), self.is_train))
            #output shape: [224, 224, 32]
            with tf.variable_scope('cw_conv_56'):
                gen_56 = conv2d(dconv5, 3, 'pw_conv', kernel_size=4, strides = 1,
                                 activation = tf.nn.tanh)
            with tf.variable_scope('cw_conv_112'):
                gen_112 = conv2d(dconv6, 3, 'pw_conv', kernel_size=4, strides = 1,
                                 activation = tf.nn.tanh)
            with tf.variable_scope('cw_conv_224'):
                gen_224 = conv2d(dconv7, 3, 'pw_conv', kernel_size=4, strides = 1,
                                 activation = tf.nn.tanh)
        
            return (gen_56+1)*127.5, (gen_112+1)*127.5, (gen_224+1)*127.5
        
    def discriminator(self, images_56, images_112, images_224, y=None, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse) as scope:
            # 70 x 70 patchGAN: 
            # shape of input images 224 x 224 x 3
            # shape of input images 224 x 224 x 6, concat profile and front face
            with tf.variable_scope("dis_0"):
                d0_bn1 = batch_norm(name='d_bn1')
                d0_bn2 = batch_norm(name='d_bn2')
                d0_bn3 = batch_norm(name='d_bn3')
                d0_bn4 = batch_norm(name='d_bn4')
                
                images_224 = images_224 / 127.5 - 1
                with tf.variable_scope('d_conv0'):
                    d0_h0 = lrelu(conv2d(images_224, 32, 'd_conv0', kernel_size=4, strides=2))
                # h0 is (112 x 112 x 32)
                with tf.variable_scope('d_conv1'):
                    d0_h1 = lrelu(d0_bn1(conv2d(d0_h0, 64, 'd_conv1', kernel_size=4, strides=2), self.is_train))
                # h1 is (56 x 56 x 64)
                with tf.variable_scope('d_conv2'):
                    d0_h2 = lrelu(d0_bn2(conv2d(d0_h1, 128, 'd_conv2', kernel_size=4, strides=2), self.is_train))
                # h2 is (28 x 28 x 128)
                with tf.variable_scope('d_conv3'):
                    d0_h3 = lrelu(d0_bn3(conv2d(d0_h2, 256, 'd_conv3', kernel_size=4, strides=2), self.is_train))
                # h3 is (14 x 14 x 256)
                with tf.variable_scope('d_conv4'):
                    d0_h4 = lrelu(d0_bn4(conv2d(d0_h3, 256, 'd_conv4', kernel_size=4, strides=2), self.is_train))
                # h4 is (7 x 7 x 256)
                #dim_wh, dim_c = h2.get_shape().as_list()[1], h2.get_shape().as_list()[3]
                #h5 = fullyConnect(tf.reshape(h4, [cfg.batch_size, -1]), 1, 'd_fc1')
                with tf.variable_scope('d_conv5'):
                    d0_h5 = conv2d(d0_h4, 1, 'd_conv5', kernel_size=4, strides=2)
                # h5 is (4 x 4 x 1)
            with tf.variable_scope("dis_1"):
                d1_bn1 = batch_norm(name='d_bn1')
                d1_bn2 = batch_norm(name='d_bn2')
                d1_bn3 = batch_norm(name='d_bn3')
                
                images_112 = images_112 / 127.5 - 1
                with tf.variable_scope('d_conv0'):
                    d1_h0 = lrelu(conv2d(images_112, 32, 'd_conv0', kernel_size=4, strides=2))
                # h0 is (56 x 56 x 32)
                with tf.variable_scope('d_conv1'):
                    d1_h1 = lrelu(d1_bn1(conv2d(d1_h0, 64, 'd_conv1', kernel_size=4, strides=2), self.is_train))
                # h1 is (28 x 28 x 64)
                with tf.variable_scope('d_conv2'):
                    d1_h2 = lrelu(d1_bn2(conv2d(d1_h1, 128, 'd_conv2', kernel_size=4, strides=2), self.is_train))
                # h2 is (14 x 14 x 128)
                with tf.variable_scope('d_conv3'):
                    d1_h3 = lrelu(d1_bn3(conv2d(d1_h2, 256, 'd_conv3', kernel_size=4, strides=2), self.is_train))
                # h3 is (7 x 7 x 256)
                with tf.variable_scope('d_conv4'):
                    d1_h4 = conv2d(d1_h3, 1, 'd_conv4', kernel_size=4, strides=2)
                # h4 is (4 x 4 x 1)
            with tf.variable_scope("dis_2"):
                d2_bn1 = batch_norm(name='d_bn1')
                d2_bn2 = batch_norm(name='d_bn2')
                
                images_56 = images_56 / 127.5 - 1
                with tf.variable_scope('d_conv0'):
                    d2_h0 = lrelu(conv2d(images_56, 32, 'd_conv0', kernel_size=4, strides=2))
                # h0 is (28 x 28 x 32)
                with tf.variable_scope('d_conv1'):
                    d2_h1 = lrelu(d2_bn1(conv2d(d2_h0, 64, 'd_conv1', kernel_size=4, strides=2), self.is_train))
                # h1 is (14 x 14 x 64)
                with tf.variable_scope('d_conv2'):
                    d2_h2 = lrelu(d2_bn2(conv2d(d2_h1, 128, 'd_conv2', kernel_size=4, strides=2), self.is_train))
                # h2 is (7 x 7 x 128)
                with tf.variable_scope('d_conv3'):
                    d2_h3 = conv2d(d2_h2, 1, 'd_conv3', kernel_size=4, strides=2)
                # h3 is (4 x 4 x 1)
            
            return tf.nn.sigmoid(d2_h3), d2_h3, tf.nn.sigmoid(d1_h4), d1_h4, tf.nn.sigmoid(d0_h5), d0_h5

    def loss(self):
        with tf.name_scope('loss') as scope:
            # 1. Frontalization Loss: L1-Norm
            #face_mask = Image.open('tt.bmp').crop([13,13,237,237])
            #face_mask = np.array(face_mask, dtype=np.float32).reshape(224,224,1) / 255.0
            #face_mask = np.tile(face_mask, [cfg.batch_size, 1, 1, 3])
            #self.front_loss = tf.losses.absolute_difference(labels=self.front, 
            #                                                predictions=self.texture)
            self.front_loss = tf.reduce_mean(tf.abs(self.front/255. - self.texture_224/255.)) # 
            tf.add_to_collection('losses', self.front_loss)
            
            # 2. Feature Loss: Cosine-Norm / L2-Norm
            enc_fea_recon_norm = self.enc_fea_recon / tf.norm(self.enc_fea_recon,
                axis=1,keep_dims=True)
            enc_fea_recon_gt_norm = self.enc_fea / tf.norm(self.enc_fea,
                axis=1,keep_dims=True)
            #self.feature_loss = tf.losses.cosine_distance(labels=enc_fea_recon_gt_norm,
            #    predictions=enc_fea_recon_norm, dim=1)
            self.feature_loss = tf.losses.mean_squared_error(labels=self.enc_fea_recon,
                                                             predictions=self.enc_fea) #
            tf.add_to_collection('losses', self.feature_loss)
            
            # 3. L2 Regulation Loss
            self.l2_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            tf.add_to_collection('losses', self.l2_loss)
            
            # 4. Adversarial Loss
            self.d_loss_real_56 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dr_56_logits, 
                                                                                 labels=tf.ones_like(self.dr_56)))
            self.d_loss_fake_56 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.df_56_logits, 
                                                                                 labels=tf.zeros_like(self.df_56)))
            self.d_loss_real_112 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dr_112_logits, 
                                                                                 labels=tf.ones_like(self.dr_112)))
            self.d_loss_fake_112 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.df_112_logits, 
                                                                                 labels=tf.zeros_like(self.df_112)))
            self.d_loss_real_224 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dr_224_logits, 
                                                                                 labels=tf.ones_like(self.dr_224)))
            self.d_loss_fake_224 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.df_224_logits, 
                                                                                 labels=tf.zeros_like(self.df_224)))
            #self.d_loss = (self.d_loss_real_56 + self.d_loss_fake_56 + 
            #               self.d_loss_real_112 + self.d_loss_fake_112 + 
            #               self.d_loss_real_224 + self.d_loss_fake_224) / 6 #
            self.d_loss = self.d_loss_real_224 + self.d_loss_fake_224
            self.g_loss_56 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.df_56_logits, 
                                                                                 labels=tf.ones_like(self.df_56)))
            self.g_loss_112 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.df_112_logits, 
                                                                                 labels=tf.ones_like(self.df_112)))
            self.g_loss_224 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.df_224_logits, 
                                                                                 labels=tf.ones_like(self.df_224)))
            #self.g_loss = (self.g_loss_56 + self.g_loss_112 + self.g_loss_224) / 3
            self.g_loss = self.g_loss_224
            tf.add_to_collection('losses', self.d_loss)
            tf.add_to_collection('losses', self.g_loss)
            
            # 5. Total Loss
            self.texture_loss = cfg.lambda_l1 * self.front_loss + cfg.lambda_fea * self.feature_loss + \
                                cfg.lambda_reg * self.l2_loss
            self.gen_loss = cfg.lambda_fea * self.feature_loss + \
                            cfg.lambda_gan * self.g_loss + cfg.lambda_reg * self.l2_loss
            self.dis_loss = cfg.lambda_gan * self.d_loss + cfg.lambda_reg * self.l2_loss

    # Summary
    def _summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/front_loss', self.front_loss))
        train_summary.append(tf.summary.scalar('train/feature_loss', self.feature_loss))
        self.train_summary = tf.summary.merge(train_summary)
        
        #correct_prediction = tf.equal(tf.to_int32(self.labels), self.argmax_idx)
        #self.batch_accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        #self.test_acc = tf.placeholder_with_default(tf.constant(0.), shape=[])
        
    def decoder_bn(self, feature, y=None, reuse=False):
        '''
        decoder of generator, decoder feature from vgg
        input: face identity feature from VGG-16 / VGG-res50
        return: generated front face [0, 255].
        '''
        # The feature vector extract from profile by VGG-16 is 4096-D
        # The feature vector extract from profile by Resnet-50 is 2048-D
        with tf.variable_scope('decoder', reuse=reuse) as scope:
            # Construct BatchNorm Layer
            #bn0 = batch_norm(name='bn0')
            #bn0_2 = batch_norm(name='bn0_2')
            bn1_1 = batch_norm(name='bn1_1')
            bn1_2 = batch_norm(name='bn1_2')
            bn2_1 = batch_norm(name='bn2_1')
            bn2_2 = batch_norm(name='bn2_2')
            bn3_1 = batch_norm(name='bn3_1')
            bn3_2 = batch_norm(name='bn3_2')
            
            # map from fc7_encoder to 14 × 14 × 256 localized features
            fc1 = fullyConnect(feature, 7*7*256, 'fc1') # bn0()
            fc1_reshape = tf.reshape(fc1, [-1,7,7,256])

            # Stacked Transpose Convolutions:(2048)
            #g_input = tf.reshape(feature, [-1, 1, 1, 2048])
            #with tf.variable_scope('dconv0_1'):
            #    dconv0_1 = tf.nn.relu(deconv2d(g_input, 512, 'dconv0_1', 
            #                          kernel_size=4, strides = 1, padding='valid'))
            #ouput shape: [4, 4, 512]
            #with tf.variable_scope('dconv0_2'):
            #    dconv0_2 = tf.nn.relu(bn0_2(deconv2d(dconv0_1, 512, 'dconv0_2', 
            #                          kernel_size=4, strides = 1, padding='valid'), self.is_train))
            #input shape: [7, 7, 512]
            #res1 = res_block(dconv0_2, 'res1', self.is_train)
            #output shape: [7, 7, 512]
            #res2 = res_block(res1, 'res2', self.is_train)
            #output shape: [7, 7, 256]
            with tf.variable_scope('dconv1_1'):
                dconv1_1 = tf.nn.relu(bn1_1(deconv2d(fc1_reshape, 256, 'dconv1_1', 
                                      kernel_size=4, strides = 2), self.is_train))
            #ouput shape: [14, 14, 256]
            with tf.variable_scope('dconv1_2'):
                dconv1_2 = tf.nn.relu(bn1_2(deconv2d(dconv1_1, 128, 'dconv1_2', 
                                      kernel_size=4, strides = 2), self.is_train))
            #output shape: [28, 28, 128]
            with tf.variable_scope('dconv2_1'):
                dconv2_1 = tf.nn.relu(bn2_1(deconv2d(dconv1_2, 64, 'dconv2_1', 
                                      kernel_size=4, strides = 2), self.is_train))
            #output shape: [56, 56, 64]
            with tf.variable_scope('dconv2_2'):
                dconv2_2 = tf.nn.relu(bn2_2(deconv2d(dconv2_1, 32, 'dconv2_2', 
                                      kernel_size=4, strides = 2), self.is_train))
            #output shape: [112, 112, 32]
            with tf.variable_scope('dconv3_1'):
                dconv3_1 = tf.nn.relu(bn3_1(deconv2d(dconv2_2, 32, 'dconv3_1', 
                                      kernel_size=4, strides = 2), self.is_train))
            #output shape: [224, 224, 32]
            with tf.variable_scope('pw_conv'):
                pw_conv = conv2d(dconv3_1, 3, 'pw_conv', kernel_size=1, strides = 1,
                                 activation = tf.nn.tanh)
            texture = (pw_conv + 1) * 127.5
        
            return texture
        
    def discriminator_bn(self, images, y=None, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse) as scope:
            # 70 x 70 patchGAN: 
            # shape of input images 224 x 224 x 6, concat profile and front face
            # shape of input images 7 x 7 x 1024,
            d_bn1 = batch_norm(name='d_bn1')
            d_bn2 = batch_norm(name='d_bn2')
            d_bn3 = batch_norm(name='d_bn3')
            d_bn4 = batch_norm(name='d_bn4')
            
            images = images / 127.5 - 1
            with tf.variable_scope('d_conv0'):
                h0 = lrelu(conv2d(images, 32, 'd_conv0', kernel_size=4, strides=2))
            # h0 is (112 x 112 x 32)
            
            with tf.variable_scope('d_conv1'):
                h1 = lrelu(d_bn1(conv2d(h0, 64, 'd_conv1', kernel_size=4, strides=2), self.is_train))
            # h1 is (56 x 56 x 64)
            with tf.variable_scope('d_conv2'):
                h2 = lrelu(d_bn2(conv2d(h1, 128, 'd_conv2', kernel_size=4, strides=2), self.is_train))
            # h2 is (28 x 28 x 128)
            with tf.variable_scope('d_conv3'):
                h3 = lrelu(d_bn3(conv2d(h2, 256, 'd_conv3', kernel_size=4, strides=2), self.is_train))
            # h3 is (14 x 14 x 256)
            with tf.variable_scope('d_conv4'):
                h4 = lrelu(d_bn4(conv2d(h3, 256, 'd_conv4', kernel_size=4, strides=2), self.is_train))
            # h4 is (7 x 7 x 256)
            #dim_wh, dim_c = h2.get_shape().as_list()[1], h2.get_shape().as_list()[3]
            #h5 = fullyConnect(tf.reshape(h4, [cfg.batch_size, -1]), 1, 'd_fc1')
            with tf.variable_scope('d_conv5'):
                h5 = conv2d(h4, 1, 'd_conv5', kernel_size=4, strides=1)
            # h5 is (7 x 7 x 1)
            
            return tf.nn.sigmoid(h5), h5
            
if '__name__' == '__main__':
    net = Net()
    net.build()
