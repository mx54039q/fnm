#coding: utf-8
import tensorflow as tf
from PIL import Image
from config import cfg
from utils import loadData
from resnet50 import Resnet50
from ops import *

epsilon = 1e-9

class LSGAN(object):
    """
    version: 
    1. 使用pipeline读取训练图片
    2. G_enc为VGG2, G_dec为全卷积网络, Pixel-Wise Normalization和ReLU, 最后一层不用PN, 上采样反卷积(k4s2),
       三个尺度输出(56/112/224), 2个尺度输出后再join初始特征并接一个残差模块, 输出接tanh并归一化到[0,255]
    3. 对应G有三个尺度的独立的判别器, 输入先进行减均值归一化, BN和LReLU, 第一层和最后一层不用BN, 全卷积网络(k4s2)
    4. G损失函数:VGG特征余弦距离/MSE距离/人脸水平对称; D损失函数:MSE距离
    5. 2个优化器Adam(beta1=0.5, beta2=0.9), lr_G=lr_D, 
    6. D的参数强制截断[-0.01,0.01], 注意是截断D所有参数
    7. 两个网络的L2规则化
    """
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.data_feed = loadData(batch_size=cfg.batch_size, train_shuffle=True) # False
            
            # Construct Template Model (G_enc) to encoder input face
            with tf.variable_scope('face_model'):
                self.face_model = Resnet50() # Vgg16()
                self.face_model.build()
                print('VGG model built successfully.')
            
            # Construct G_dec and D in 3 scale
            if cfg.is_train:                
                self.is_train = tf.placeholder(tf.bool, name='is_train')
                self.profile,self.gt,self.front,self.resized_56,self.resized_112 = self.data_feed.get_train()
                
                # Construct Model
                self.build_arch()
                print('Model built successfully.')
                
                all_vars = tf.trainable_variables()
                self.vars_gen = [var for var in all_vars if var.name.startswith('decoder')]
                self.vars_dis = [var for var in all_vars if var.name.startswith('discriminator')]
                self.loss()
                self._summary()
                
                # Trainer
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.train_wu = tf.train.AdamOptimizer(cfg.lr, beta1=cfg.beta1, beta2=cfg.beta2).minimize(
                                self.wu_loss,
                                global_step=self.global_step, var_list=self.vars_gen)
                self.train_gen = tf.train.AdamOptimizer(cfg.lr, beta1=cfg.beta1, beta2=cfg.beta2).minimize(
                                 self.gen_loss,
                                 global_step=self.global_step, var_list=self.vars_gen)
                self.train_dis = tf.train.AdamOptimizer(cfg.lr, beta1=cfg.beta1, beta2=cfg.beta2).minimize(
                                 self.dis_loss,
                                 global_step=self.global_step, var_list=self.vars_dis)
                
                # Weight Clip on D
                with tf.name_scope('clip_weightOf_D'):
                    self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.vars_dis]
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
        print 'Face model output feature shape:', self.enc_fea.get_shape()
        
        # Decoder front faces of 3 scale from vgg feature
        self.texture_56, self.texture_112, self.texture_224 = self.decoder(self.enc_fea)
        assert self.texture_224.get_shape().as_list()[1:] == [224,224,3]
        
        # Map texture into features again by VGG    
        with tf.name_scope('encoder_recon'):
            _, self.enc_fea_recon = self.face_model.forward(self.texture_224)
        assert self.enc_fea_recon.get_shape().as_list()[1:] == [2048]
        
        # Construct discriminator between generalized front face and ground truth
        self.dr_56, self.dr_56_logits,self.dr_112, self.dr_112_logits,self.dr_224, self.dr_224_logits = \
            self.discriminator(self.resized_56,self.resized_112,self.front) #
        self.df_56, self.df_56_logits,self.df_112, self.df_112_logits,self.df_224, self.df_224_logits = \
            self.discriminator(self.texture_56,self.texture_112,self.texture_224, reuse=True)
        
    def decoder(self, feature, reuse=False):
        """
        decoder of generator, decoder feature from vgg
        args: 
            feature: face identity feature from VGG-16 / VGG-res50.
            reuse: Whether to reuse the model(Default False).
        return: 
            3 scale generated front face in [0, 255].
        """
        # The feature vector extract from profile by VGG-16 is 4096-D
        # The feature vector extract from profile by Resnet-50 is 2048-D
        with tf.variable_scope('decoder', reuse=reuse) as scope:        
            # Stacked Transpose Convolutions:(2048)
            g_input = tf.reshape(feature, [-1, 1, 1, 2048])
            with tf.variable_scope('dconv0'):
                dconv0 = tf.nn.relu(ins_norm(deconv2d(g_input, 512, 'dconv0', 
                                    kernel_size=4, strides = 1, padding='valid')))
            #ouput shape: [4, 4, 512]
            with tf.variable_scope('dconv1'):
                dconv1 = tf.nn.relu(ins_norm(deconv2d(dconv0, 512, 'dconv1', 
                                        kernel_size=4, strides = 1, padding='valid')))
            #input shape: [7, 7, 512]
            with tf.variable_scope('dconv3'):
                dconv3 = tf.nn.relu(ins_norm(deconv2d(dconv1, 256, 'dconv3', 
                                        kernel_size=4, strides = 2)))
            #ouput shape: [14, 14, 256]
            with tf.variable_scope('dconv4'):
                dconv4 = tf.nn.relu(ins_norm(deconv2d(dconv3, 128, 'dconv4', 
                                        kernel_size=4, strides = 2)))
            #output shape: [28, 28, 128]
            with tf.variable_scope('dconv5'):
                dconv5 = tf.nn.relu(ins_norm(deconv2d(dconv4, 64, 'dconv5', 
                                        kernel_size=4, strides = 2)))
            #output shape: [56, 56, 64]
            with tf.variable_scope('dconv5_concat'):
                fc1 = tf.nn.relu(fullyConnect(feature, 56 * 56, 'fc1'))
                fc1 = tf.reshape(fc1, [-1,56,56,1])
                fc1 = tf.tile(fc1, [1, 1, 1, 16])
                dconv5_concat = tf.concat([dconv5, fc1], 3)
            #output shape: [56, 56, 80]
            with tf.variable_scope('res1'):
                res1_conv1 = tf.nn.relu(ins_norm(conv2d(dconv5_concat, 128, 'conv1', 
                                        kernel_size=4, strides = 1)))
                res1_conv2 = ins_norm(conv2d(res1_conv1, 80, 'conv2', 
                                        kernel_size=4, strides = 1))
                res1 = tf.nn.relu(tf.add(dconv5_concat, res1_conv2))
            #output shape: [56, 56, 80]
            with tf.variable_scope('dconv6'):
                dconv6 = tf.nn.relu(ins_norm(deconv2d(res1, 32, 'dconv6', 
                                        kernel_size=4, strides = 2)))
            #output shape: [112, 112, 32]
            with tf.variable_scope('dconv6_concat'):
                fc2 = tf.nn.relu(fullyConnect(feature, 112 * 112, 'fc2'))
                fc2 = tf.reshape(fc2, [-1,112,112,1])
                fc2 = tf.tile(fc2, [1, 1, 1, 8])
                dconv6_concat = tf.concat([dconv6, fc2], 3)
            #output shape: [112, 112, 40]
            with tf.variable_scope('res2'):
                res2_conv1 = tf.nn.relu(ins_norm(conv2d(dconv6_concat, 64, 'conv1', 
                                        kernel_size=4, strides = 1)))
                res2_conv2 = ins_norm(conv2d(res2_conv1, 40, 'conv2', 
                                        kernel_size=4, strides = 1))
                res2 = tf.nn.relu(tf.add(dconv6_concat, res2_conv2))
            #input shape: [112, 112, 40]
            with tf.variable_scope('dconv7'):
                dconv7 = tf.nn.relu(ins_norm(deconv2d(res2, 32, 'dconv7', 
                                        kernel_size=4, strides = 2)))
            #output shape: [224, 224, 32]
            with tf.variable_scope('cw_conv_56'):
                gen_56 = tf.nn.tanh(conv2d(dconv5, 3, 'pw_conv', kernel_size=4, strides = 1,))
            with tf.variable_scope('cw_conv_112'):
                gen_112 = tf.nn.tanh(conv2d(dconv6, 3, 'pw_conv', kernel_size=4, strides = 1,))
            with tf.variable_scope('cw_conv_224'):
                gen_224 = tf.nn.tanh(conv2d(dconv7, 3, 'pw_conv', kernel_size=4, strides = 1,))
        
            return (gen_56+1)*127.5, (gen_112+1)*127.5, (gen_224+1)*127.5
        
    def discriminator(self, images_56, images_112, images_224, reuse=False):
        """
        patch GAN, output logits shape [bs, 4, 4, 1]
        args: 
            image_56: front face in [0,255]. [56,56,3]
            image_112: front face in [0,255]. [112,112,3]
            image_224: front face in [0,255]. [224,224,3]
            reuse: Whether to reuse the model(Default False).
        return: 
            3 pairs of sidmoid(logits) and logits.
        """
        with tf.variable_scope("discriminator", reuse=reuse) as scope:
            with tf.variable_scope("dis_0"):
                d0_bn1 = batch_norm(name='bn1')
                d0_bn2 = batch_norm(name='bn2')
                d0_bn3 = batch_norm(name='bn3')
                d0_bn4 = batch_norm(name='bn4')
                d0_bn5 = batch_norm(name='bn5')
                
                images_224 = images_224 / 127.5 - 1
                with tf.variable_scope('d_conv0'):
                    d0_h0 = lrelu(conv2d(images_224, 32, 'd_conv0', kernel_size=4, strides=2))
                # h0 is (112 x 112 x 32)
                with tf.variable_scope('d_conv1'):
                    d0_h1 = lrelu(d0_bn1(conv2d(d0_h0, 64, 'd_conv1', kernel_size=4, strides=2)))
                # h1 is (56 x 56 x 64)
                with tf.variable_scope('d_conv2'):
                    d0_h2 = lrelu(d0_bn2(conv2d(d0_h1, 128, 'd_conv2', kernel_size=4, strides=2)))
                # h2 is (28 x 28 x 128)
                with tf.variable_scope('d_conv3'):
                    d0_h3 = lrelu(d0_bn3(conv2d(d0_h2, 256, 'd_conv3', kernel_size=4, strides=2)))
                # h3 is (14 x 14 x 256)
                with tf.variable_scope('d_conv4'):
                    d0_h4 = lrelu(d0_bn4(conv2d(d0_h3, 256, 'd_conv4', kernel_size=4, strides=2)))
                # h4 is (7 x 7 x 256)
                with tf.variable_scope('d_conv5'):
                    d0_h5 = lrelu(d0_bn5(conv2d(d0_h4, 256, 'd_conv5', kernel_size=4, strides=2)))
                # h5 is (4 x 4 x 256)
                d0_h6 = conv2d(d0_h5, 1, 'd_conv6', kernel_size=4, strides=1, padding='valid')
                # h6 is (1 x 1 x 1)
                
            with tf.variable_scope("dis_1"):
                d1_bn1 = batch_norm(name='bn1')
                d1_bn2 = batch_norm(name='bn2')
                d1_bn3 = batch_norm(name='bn3')
                d1_bn4 = batch_norm(name='bn4')
                
                images_112 = images_112 / 127.5 - 1
                with tf.variable_scope('d_conv0'):
                    d1_h0 = lrelu(conv2d(images_112, 32, 'd_conv0', kernel_size=4, strides=2))
                # h0 is (56 x 56 x 32)
                with tf.variable_scope('d_conv1'):
                    d1_h1 = lrelu(d1_bn1(conv2d(d1_h0, 64, 'd_conv1', kernel_size=4, strides=2)))
                # h1 is (28 x 28 x 64)
                with tf.variable_scope('d_conv2'):
                    d1_h2 = lrelu(d1_bn2(conv2d(d1_h1, 128, 'd_conv2', kernel_size=4, strides=2)))
                # h2 is (14 x 14 x 128)
                with tf.variable_scope('d_conv3'):
                    d1_h3 = lrelu(d1_bn3(conv2d(d1_h2, 256, 'd_conv3', kernel_size=4, strides=2)))
                # h3 is (7 x 7 x 256)
                with tf.variable_scope('d_conv4'):
                    d1_h4 = lrelu(d1_bn4(conv2d(d1_h3, 256, 'd_conv4', kernel_size=4, strides=2)))
                # h4 is (4 x 4 x 256)
                d1_h5 = conv2d(d1_h4, 1, 'd_conv5', kernel_size=4, strides=1, padding='valid')
                # h5 is (1 x 1 x 1)
                
            with tf.variable_scope("dis_2"):
                d2_bn1 = batch_norm(name='bn1')
                d2_bn2 = batch_norm(name='bn2')
                d2_bn3 = batch_norm(name='bn3')
                
                images_56 = images_56 / 127.5 - 1
                with tf.variable_scope('d_conv0'):
                    d2_h0 = lrelu(conv2d(images_56, 32, 'd_conv0', kernel_size=4, strides=2))
                # h0 is (28 x 28 x 32)
                with tf.variable_scope('d_conv1'):
                    d2_h1 = lrelu(d2_bn1(conv2d(d2_h0, 64, 'd_conv1', kernel_size=4, strides=2)))
                # h1 is (14 x 14 x 64)
                with tf.variable_scope('d_conv2'):
                    d2_h2 = lrelu(d2_bn2(conv2d(d2_h1, 128, 'd_conv2', kernel_size=4, strides=2)))
                # h2 is (7 x 7 x 128)
                with tf.variable_scope('d_conv3'):
                    d2_h3 = lrelu(d2_bn3(conv2d(d2_h2, 256, 'd_conv3', kernel_size=4, strides=2)))
                # h3 is (4 x 4 x 256)
                d2_h4 = conv2d(d2_h3, 1, 'd_conv4', kernel_size=4, strides=1, padding='valid')
                # h4 is (1 x 1 x 1)
                
        return tf.nn.sigmoid(d2_h4), d2_h4, tf.nn.sigmoid(d1_h5), d1_h5, tf.nn.sigmoid(d0_h6), d0_h6

    def loss(self):
        """
        Loss Functions
        """
        with tf.name_scope('loss') as scope:
            # 1. Frontalization Loss: L1-Norm
            with tf.name_scope('Pixel_Loss'):
                #face_mask = Image.open('tt.bmp').crop([13,13,237,237])
                #face_mask = np.array(face_mask, dtype=np.float32).reshape(224,224,1) / 255.0
                #face_mask = np.tile(face_mask, [cfg.batch_size, 1, 1, 3])
                #self.front_loss = tf.losses.absolute_difference(labels=self.front, 
                #                                                predictions=self.texture)
                self.front_loss = tf.reduce_mean(tf.abs(self.gt/255. - self.texture_224/255.))
                tf.add_to_collection('losses', self.front_loss)
            
            # 2. Feature Loss: Cosine-Norm / L2-Norm
            with tf.name_scope('Perceptual_Loss'):
                enc_fea_recon_norm = self.enc_fea_recon / tf.norm(self.enc_fea_recon,
                    axis=1,keep_dims=True)
                enc_fea_recon_gt_norm = self.enc_fea / tf.norm(self.enc_fea,
                    axis=1,keep_dims=True)
                self.feature_loss = tf.losses.cosine_distance(labels=enc_fea_recon_gt_norm,
                    predictions=enc_fea_recon_norm, dim=1)
                #self.feature_loss = tf.losses.mean_squared_error(labels=self.enc_fea_recon,
                #                                                 predictions=self.enc_fea) #
                tf.add_to_collection('losses', self.feature_loss)
            
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
                self.d_loss = (tf.losses.mean_squared_error(tf.ones_like(self.dr_56_logits), self.dr_56_logits) + 
                               tf.losses.mean_squared_error(tf.zeros_like(self.df_56_logits), self.df_56_logits) +
                               tf.losses.mean_squared_error(tf.ones_like(self.dr_112_logits), self.dr_112_logits) +
                               tf.losses.mean_squared_error(tf.zeros_like(self.df_112_logits), self.df_112_logits) +
                               tf.losses.mean_squared_error(tf.ones_like(self.dr_224_logits), self.dr_224_logits) +
                               tf.losses.mean_squared_error(tf.zeros_like(self.df_224_logits), self.df_224_logits)) / 6
                self.g_loss = (tf.losses.mean_squared_error(tf.ones_like(self.df_56_logits), self.df_56_logits) + 
                               tf.losses.mean_squared_error(tf.ones_like(self.df_112_logits), self.df_112_logits) + 
                               tf.losses.mean_squared_error(tf.ones_like(self.df_224_logits), self.df_224_logits)) / 3
                tf.add_to_collection('losses', self.d_loss)
                tf.add_to_collection('losses', self.g_loss)
            
            # 5. Symmetric Loss
            with tf.name_scope('Symmetric_Loss'):
                mirror_image = tf.reverse(self.texture_224, axis=[2])
                self.sym_loss = tf.reduce_mean(tf.abs(mirror_image/255. - self.texture_224/255.))
           
            # 6. Total Loss
            with tf.name_scope('Total_Loss'):
                self.wu_loss = cfg.lambda_l1 * self.front_loss + cfg.lambda_fea * self.feature_loss + \
                               cfg.lambda_sym * self.sym_loss + self.reg_gen
                self.gen_loss = cfg.lambda_l1 * self.front_loss + cfg.lambda_fea * self.feature_loss + \
                                cfg.lambda_gan * self.g_loss + cfg.lambda_sym * self.sym_loss + self.reg_gen
                self.dis_loss = cfg.lambda_gan * self.d_loss + self.reg_dis
                
    def _summary(self):
        """
        Tensorflow Summary
        """
        train_summary = []
        train_summary.append(tf.summary.scalar('train/front_loss', self.front_loss))
        train_summary.append(tf.summary.scalar('train/feature_loss', self.feature_loss))
        self.train_summary = tf.summary.merge(train_summary)
        
        #correct_prediction = tf.equal(tf.to_int32(self.labels), self.argmax_idx)
        #self.batch_accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        #self.test_acc = tf.placeholder_with_default(tf.constant(0.), shape=[])
    
if '__name__' == '__main__':
    net = LSGAN()
    net.build()
