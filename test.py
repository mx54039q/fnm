#coding: utf-8
import os
import tensorflow as tf
from PIL import Image
from WGAN_GP import WGAN_GP
from config import cfg
from utils import loadData
import numpy as np

"""
usage: python test.py --test_path ../../dataset/saveRejectFace --test_list ../fnm/mpie/school_profile.txt
"""


def read_img(path, img):
  '''Read test images'''
  #img_array = np.array(Image.open(os.path.join(path, img)), dtype=np.float32)
  img_array = np.array(Image.open(os.path.join(path, img)).resize((224,224)), dtype=np.float32)
  img_array = np.expand_dims(img_array, axis=0)
  return img_array

def save_img(path, img, img_array, img2):
  '''Save test images'''
  save_path = os.path.join(path, img)
  img_array = np.squeeze(np.concatenate((img_array, img2), 2).astype(np.uint8))
  Image.fromarray(img_array).save(save_path)

def main(_):
  if not os.path.exists('./test'):
    os.makedirs('./test')
   
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device_id
  
  cfg.batch_size = 1
  net = WGAN_GP()
  
  profile = tf.placeholder(tf.float32, [1,224,224,3], name='profile')
  front = tf.placeholder(tf.float32, [1,224,224,3], name='profile')
  net.build_up(profile, front)
  
  print('Load Finetuned Model Successfully!')
  
  # Train or Test
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config, graph=net.graph) as sess:
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver(max_to_keep=0)  #
    saver.restore(sess, cfg.checkpoint_ft)
    
    test_list = np.loadtxt(cfg.test_list, dtype='string',delimiter=',')
    for img in test_list[:50]:
      print(img)
      img_np = read_img(cfg.test_path, img)
      img_gen = sess.run(net.gen_p, {profile:img_np, net.is_train:False}) #
      save_img('test', img, img_np, img_gen)
  
if __name__ == "__main__":
  tf.app.run()
  
