import tensorflow as tf

flags = tf.app.flags


############################
#    hyper parameters      #
############################

# For separate margin loss
flags.DEFINE_float('lambda_l1', 1, 'down weight of the loss for L1 texture loss') #
flags.DEFINE_float('lambda_fea', 0.03, 'down weight of the loss for face model feature loss') #
flags.DEFINE_float('lambda_reg', 1, 'down weight of the loss for L2 regularitaion loss') #
flags.DEFINE_float('lambda_gan', 0.01, 'down weight of the loss for gan loss') #

# For training
flags.DEFINE_integer('dataset_size', 14000, 'number of images in the dataset') # 120000
flags.DEFINE_string('data_path', '/home/ycqian/session01_align', 'dataset path') # /home/pris/Videos/session01_align
flags.DEFINE_string('train_list', 'mpie/setting1_train2.txt', 'train list') # session01_train_shuffle.txt
flags.DEFINE_string('test_list', 'mpie/session01_test2.txt', 'test set path') # lfw/lfw.txt
flags.DEFINE_boolean('is_train', True, 'train or frontalize test')
flags.DEFINE_boolean('is_finetune', False, 'finetune') # False, True
flags.DEFINE_string('logdir', 'logdir/setting1/setting1_6', 'model directory') #setting1/setting1_5
flags.DEFINE_string('summary_dir', 'log/setting1_6', 'logs directory') # setting1_5
flags.DEFINE_string('model_path', 'logdir/setting1/setting1_6-pretrain', 'finetune model path') #
flags.DEFINE_integer('batch_size', 5, 'batch size')
flags.DEFINE_integer('decay_steps', 100, 'learning rate decay steps')
flags.DEFINE_integer('epoch', 5, 'epoch')
flags.DEFINE_integer('train_sum_freq', 400, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('test_sum_freq', 30, 'the frequency of saving test summary(step)')
flags.DEFINE_integer('save_freq', 100, 'the frequency of saving model(epoch)')
flags.DEFINE_boolean('use_profile', True, 'Use profile image or profile feature') #
flags.DEFINE_boolean('mask_with_y', True, 'use the true label to mask out target capsule or not')
flags.DEFINE_boolean('crop', True, 'Crop image to target size')
flags.DEFINE_float('lr', 0.0002, 'base learning rate')
flags.DEFINE_float('beta1', 0.5, 'momentum term of adam')
flags.DEFINE_float('stddev', 0.02, 'stddev for W initializer')


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
