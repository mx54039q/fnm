import os
import tensorflow as tf
from tqdm import tqdm

from config import cfg # setting / configuration / hyper parameters
from utils import loadData # get data
from Net import Net # Class Net
net = Net(is_training=cfg.is_training) # construct Capsule Network

# Net.fc7_encoder: feature of the input images
# Net.texture: frontal result 
# Net.recon_feature: feature of the texture
# Net.recon_feature_gt: feature of the ground true front

data_feed = loadData(batch_size = cfg.batch_size)
tf.logging.info('Graph loaded')

path = cfg.results + '/accuracy.csv'
if not os.path.exists(cfg.results):
    os.mkdir(cfg.results)
elif os.path.exists(path):
    os.remove(path)

fd_results = open(path, 'w')
fd_results.write('step,front_loss,feature_loss\n')
config = tf.ConfigProto()

# Start
sess = tf.InteractiveSession(config=config, graph=net.graph)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=1)
saver.restore(sess, 'logdir/model_epoch_0001_step_2807')

tr_profile, tr_front = data_feed.get_train_batch()
global_step = sess.run(net.global_step)
l, lf1, lf2, lr = sess.run([net.total_loss,net.front_loss,net.feature_loss,net.lr],
  {net.profile:tr_profile, net.front:tr_front})
#l1, l2, images = sess.run([net.front_loss,net.feature_loss,net.texture], 
#    {net.profile:tr_profile, net.front:tr_front})
data_feed.save_images(images,3,3)





