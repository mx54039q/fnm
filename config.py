# coding: utf-8
# --------------------------------------------------------
# FNM
# Written by Yichen Qian
# --------------------------------------------------------

import tensorflow as tf

flags = tf.app.flags

############################
#    hyper parameters      #
############################

# For hyper parameters
flags.DEFINE_float('lambda_l1', 0.001, 'weight of the loss for L1 texture loss') # 0.001
flags.DEFINE_float('lambda_fea', 100, 'weight of the loss for face model feature loss')
flags.DEFINE_float('lambda_reg', 1e-5, 'weight of the loss for L2 regularitaion loss')
flags.DEFINE_float('lambda_gan', 1, 'weight of the loss for gan loss')
flags.DEFINE_float('lambda_gp', 10, 'weight of the loss for gradient penalty on parameter of D')

# For training
flags.DEFINE_integer('dataset_size', 297369, 'number of non-normal face set')
flags.DEFINE_string('profile_path', '', 'dataset path')  # casia_aligned_250_250_jpg
flags.DEFINE_string('profile_list', '', 'train profile list')
flags.DEFINE_string('front_path', '', 'front data path')
flags.DEFINE_string('front_list', '', 'train front list')
flags.DEFINE_string('test_path', '', 'front data path')
flags.DEFINE_string('test_list', '', 'train front list')
flags.DEFINE_boolean('is_train', True, 'train or test')
flags.DEFINE_boolean('is_finetune', False, 'finetune') # False, True
flags.DEFINE_string('face_model', 'resnet50.npy', 'face model path')
flags.DEFINE_string('checkpoint', 'checkpoint/fnm', 'checkpoint directory')
flags.DEFINE_string('summary_dir', 'log/fnm', 'logs directory')
flags.DEFINE_string('checkpoint_ft', 'checkpoint/fnm/ck-09', 'finetune or test checkpoint path')
flags.DEFINE_integer('batch_size', 16, 'batch size')
flags.DEFINE_integer('epoch', 10, 'epoch')
flags.DEFINE_integer('critic', 1, 'number of D training times')
flags.DEFINE_integer('save_freq', 1000, 'the frequency of saving model')
flags.DEFINE_float('lr', 1e-4, 'base learning rate')
flags.DEFINE_float('beta1', 0., 'beta1 momentum term of adam')
flags.DEFINE_float('beta2', 0.9, 'beta2 momentum term of adam')
flags.DEFINE_float('stddev', 0.02, 'stddev for W initializer')
flags.DEFINE_boolean('use_bias', False, 'whether to use bias')
flags.DEFINE_string('norm', 'bn', 'normalize function for G')
flags.DEFINE_string('results', 'results/fnm', 'path for saving results') #

############################
#   environment setting    #
############################
flags.DEFINE_string('device_id', '3,4', 'device id')
flags.DEFINE_integer('ori_height', 224, 'original height of profile images')
flags.DEFINE_integer('ori_width', 224, 'original width of profile images')
flags.DEFINE_integer('height', 224, 'height of images') # do not modified
flags.DEFINE_integer('width', 224, 'width of images') # do not modified
flags.DEFINE_integer('channel', 3, 'channel of images')
flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing examples')


cfg = tf.app.flags.FLAGS
# tf.logging.set_verbosity(tf.logging.INFO)
