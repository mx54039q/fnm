import tensorflow as tf

flags = tf.app.flags

############################
#    hyper parameters      #
############################

# For hyper parameters
flags.DEFINE_float('lambda_l1', 0.001, 'weight of the loss for L1 texture loss') # 0.01
flags.DEFINE_float('lambda_fea', 500, 'weight of the loss for face model feature loss') # 250
flags.DEFINE_float('lambda_reg', 1e-6, 'weight of the loss for L2 regularitaion loss') # 1e-5
flags.DEFINE_float('lambda_gan', 1, 'weight of the loss for gan loss') # 1
flags.DEFINE_float('lambda_sym', 0., 'weight of the loss for gan loss') #
flags.DEFINE_float('lambda_gp', 10, 'weight of the loss for gradient penalty on parameter of D') # 10
flags.DEFINE_float('lambda_dr', 0., 'weight of the L2 loss for the output of D according to paper') #

# For training
flags.DEFINE_integer('dataset_size', 297369, 'number of images in the dataset')
flags.DEFINE_string('profile_path', '/home/ycqian/casia_aligned_250_250_jpg', 'dataset path')
flags.DEFINE_string('profile_list', 'mpie/casia_profile.txt', 'train profile list')
flags.DEFINE_string('gt_list', 'mpie/casia_gt.txt', 'train ground truth list')
flags.DEFINE_string('front_path', '/home/ycqian/session01_align', 'front data path') #casia_aligned_250_250_jpg
flags.DEFINE_string('front_list', 'mpie/session01_front.txt', 'train front list') #casia_front.txt
flags.DEFINE_string('test_path', '/home/ycqian/an_ijb_align_2', 'test set path') # /home/ycqian/lfw-deepfunneled
flags.DEFINE_string('test_list', 'ijba/ijba_part.txt', 'test set path') # lfw/lfw_test.txt
flags.DEFINE_boolean('is_train', True, 'train or frontalize test')
flags.DEFINE_boolean('is_finetune', True, 'finetune') # False, True
flags.DEFINE_string('logdir', 'logdir/setting1/v1/v1-2_ft', 'model directory') #setting1/setting1_5
flags.DEFINE_string('summary_dir', 'log/v1', 'logs directory') # setting1_5
flags.DEFINE_string('model_path', 'logdir/setting1/v1-2', 'finetune model path') #
flags.DEFINE_integer('batch_size', 16, 'batch size')
flags.DEFINE_integer('decay_steps', 100, 'learning rate decay steps')
flags.DEFINE_integer('epoch', 5, 'epoch')
flags.DEFINE_integer('critic', 1, 'number of D training times')
flags.DEFINE_integer('train_sum_freq', 400, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('test_sum_freq', 4000, 'the frequency of saving test summary(step)')
flags.DEFINE_integer('save_freq', 500, 'the frequency of saving model')
flags.DEFINE_boolean('crop', False, 'Crop image to target size') # 
flags.DEFINE_float('lr', 1e-5, 'base learning rate') # 1e-4
flags.DEFINE_float('beta1', 0., 'beta1 momentum term of adam')
flags.DEFINE_float('beta2', 0.9, 'beta2 momentum term of adam')
flags.DEFINE_float('stddev', 0.02, 'stddev for W initializer')
flags.DEFINE_boolean('use_bias', False, 'whether to use bias')
flags.DEFINE_string('norm', 'bn', 'normalize function for G') #
flags.DEFINE_float('w_f', 0.5, 'weight of front loss for VGG-FACE') #

############################
#   environment setting    #
############################
flags.DEFINE_integer('ori_height', 250, 'original height of images')
flags.DEFINE_integer('ori_width', 250, 'original width of images')
flags.DEFINE_integer('height', 224, 'height of images')
flags.DEFINE_integer('width', 224, 'width of images')
flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing examples')
flags.DEFINE_string('results', 'results', 'path for saving results')

############################
#   distributed setting    #
############################
flags.DEFINE_integer('num_gpu', 1, 'number of gpus for distributed training')
flags.DEFINE_integer('batch_size_per_gpu', 100, 'batch size on 1 gpu')
flags.DEFINE_integer('thread_per_gpu', 8, 'Number of preprocessing threads per tower.')

cfg = tf.app.flags.FLAGS
# tf.logging.set_verbosity(tf.logging.INFO)
