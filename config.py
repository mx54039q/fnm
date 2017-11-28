import tensorflow as tf

flags = tf.app.flags


############################
#    hyper parameters      #
############################

# For separate margin loss
flags.DEFINE_float('lambda_val', 1, 'down weight of the loss for absent digit classes')

# For training
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('epoch', 50, 'epoch')
flags.DEFINE_boolean('use_profile', False, 'Use profile image or profile feature')
flags.DEFINE_boolean('mask_with_y', True, 'use the true label to mask out target capsule or not')
flags.DEFINE_boolean('crop', True, 'Crop image to target size')
flags.DEFINE_float('lr', 0.01, 'base learning rate')
flags.DEFINE_float('stddev', 0.01, 'stddev for W initializer')

############################
#   environment setting    #
############################
flags.DEFINE_integer('dataset_size', 120000, 'number of images in the dataset')
flags.DEFINE_integer('ori_height', 250, 'original height of images')
flags.DEFINE_integer('ori_width', 250, 'original width of images')
flags.DEFINE_integer('height', 224, 'height of images')
flags.DEFINE_integer('width', 224, 'width of images')
flags.DEFINE_string('data_path', '/home/pris/Videos/session01_align', 'dataset path')
flags.DEFINE_boolean('is_training', True, 'train or frontalize test')
flags.DEFINE_boolean('is_fineture', False, 'fineture')
flags.DEFINE_string('model_path', 'logdir/model_epoch_0001_step_2807', 'finetune model path')
flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing examples')
flags.DEFINE_string('logdir', 'logdir', 'logs directory')
flags.DEFINE_integer('train_sum_freq', 400, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('test_sum_freq', 30, 'the frequency of saving test summary(step)')
flags.DEFINE_integer('save_freq', 100, 'the frequency of saving model(epoch)')
flags.DEFINE_string('results', 'results', 'path for saving results')

############################
#   distributed setting    #
############################
flags.DEFINE_integer('num_gpu', 1, 'number of gpus for distributed training')
flags.DEFINE_integer('batch_size_per_gpu', 100, 'batch size on 1 gpu')
flags.DEFINE_integer('thread_per_gpu', 8, 'Number of preprocessing threads per tower.')

cfg = tf.app.flags.FLAGS
# tf.logging.set_verbosity(tf.logging.INFO)
