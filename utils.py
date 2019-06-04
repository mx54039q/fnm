# coding: utf-8
# --------------------------------------------------------
# FNM
# Written by Yichen Qian
# --------------------------------------------------------

import os
import scipy
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import struct

from config import cfg


class loadData(object):
  """Class for loading data.
  
  This is a class for loading data (e.g. image) to model. Train image 
  and test image can be obtained by function "get_train" and 
  function "get_test_batch" respectively.
  
  Args:
    batch_size (int): size of every train batch
    train_shuffle (bool): whether to shuffle train set.
    
  """
  
  def __init__(self, batch_size = 20, train_shuffle = True):
    self.batch_size = batch_size
    self.profile = np.loadtxt(cfg.profile_list, dtype='string', delimiter=',')
    self.front = np.loadtxt(cfg.front_list, dtype='string', delimiter=',')
    
    if(train_shuffle): 
      np.random.shuffle(self.profile)
      np.random.shuffle(self.front)
             
    self.test_list = np.loadtxt(cfg.test_list, dtype='string',delimiter=',') #
    self.test_index = 0
    
    # Crop Box: left, upper, right, lower
    self.crop_box = [(cfg.ori_width - cfg.width) / 2, (cfg.ori_height - cfg.height) / 2,
            (cfg.ori_width + cfg.width) / 2, (cfg.ori_height + cfg.height) / 2]     
    assert Image.open(os.path.join(cfg.profile_path, self.profile[0])).size == \
         (cfg.ori_width, cfg.ori_height)
  
  def get_train(self):
    """Get train images
    
    Train images will be horizontal-flipped, center-cropped and adjust brightness randomly.
    
    return:
      profile (tf.tensor): profile of identity A
      front (tf.tensor): front face of identity B
    """
    
    with tf.name_scope('data_feed'):
      profile_list = [os.path.join(cfg.profile_path,img) for img in self.profile]
      front_list = [os.path.join(cfg.front_path,img) for img in self.front]
      profile_files = tf.train.string_input_producer(profile_list, shuffle=False) #
      front_files = tf.train.string_input_producer(front_list, shuffle=False) #
      
      _, profile_value = tf.WholeFileReader().read(profile_files)
      profile_value = tf.image.decode_jpeg(profile_value, channels=cfg.channel)
      profile_value = tf.cast(profile_value, tf.float32)
      _, front_value = tf.WholeFileReader().read(front_files)
      front_value = tf.image.decode_jpeg(front_value, channels=cfg.channel)
      front_value = tf.cast(front_value, tf.float32)
      
      
      # Flip, crop and adjust brightness of  image
      profile_value = tf.image.random_brightness(profile_value, max_delta=20.)
      profile_value = tf.clip_by_value(profile_value, clip_value_min=0., clip_value_max=255.)
      profile_value = tf.image.random_flip_left_right(profile_value)
      profile_value = tf.random_crop(profile_value, [cfg.height, cfg.width, cfg.channel])
      
      front_value = tf.image.random_brightness(front_value, max_delta=20.)
      front_value = tf.clip_by_value(front_value, clip_value_min=0., clip_value_max=255.)
      # Args: [image, offset_height, offset_width, target_height, target_width]
      # front_value = tf.image.resize_images(front_value, [cfg.height, cfg.width])
      front_value = tf.image.resize_images(front_value, [cfg.height, cfg.width])
                            
      profile,front = tf.train.shuffle_batch([profile_value, front_value],
                          batch_size=self.batch_size,
                          num_threads=8,
                          capacity=32 * self.batch_size,
                          min_after_dequeue=self.batch_size * 16,
                          allow_smaller_final_batch=False)
      return profile, front
    
  def get_train_batch(self):
    """Get train images by preload
    
    return:
      trX: training profile images
      trY: training front images
    """
    trX = np.zeros((self.batch_size, cfg.height, cfg.width, cfg.channel), dtype=np.float32)
    trY = np.zeros((self.batch_size, cfg.height, cfg.width, cfg.channel), dtype=np.float32)
    for i in range(self.batch_size):
      try:
        trX[i] = self.read_image(self.profile[i + self.train_index], flip=True)
        trY[i] = self.read_image(self.front[i + self.train_index], flip=True)
      except:
        self.train_index = -i
        trX[i] = self.read_image(self.profile[i +self.train_index], flip=True)
        trY[i] = self.read_image(self.front[i +self.train_index], flip=True)
    self.train_index += self.batch_size
    return trX, trY
    
  def get_test_batch(self, batch_size = cfg.batch_size):
    """Get test images by batch
    
    args:
      batch_size: size of test scratch
    return:
      teX: testing profile images
      teY: testing front images, same as profile images
    """
    teX = np.zeros((batch_size, cfg.height, cfg.width, cfg.channel), dtype=np.float32)
    teY = np.zeros((batch_size, cfg.height, cfg.width, cfg.channel), dtype=np.float32)
    for i in range(batch_size):
      try:
        teX[i] = self.read_image(os.path.join(cfg.test_path,self.test_list[i +self.test_index]))
        teY[i] = self.read_image(os.path.join(cfg.test_path,self.test_list[i +self.test_index]))
      except:
        print("Test Loop at %d!" % self.test_index)
        self.test_index = -i
        teX[i] = self.read_image(os.path.join(cfg.test_path,self.test_list[i +self.test_index]))
        teY[i] = self.read_image(os.path.join(cfg.test_path,self.test_list[i +self.test_index]))
    self.test_index += batch_size
    return teX, teY

  def read_image(self, img, flip=False):
    """Read image
    
    Read a image from image path, and crop to target size
    and random flip horizontally
    
    args:
      img: image path
    return:
      img: data matrix from image
    """
    img = Image.open(img)
    if(img.mode=='L' and cfg.channel == 3):
      img = img.convert('RGB')
    if flip and np.random.random() > 0.5:
      img = img.transpose(Image.FLIP_LEFT_RIGHT)
    #if cfg.crop:
    #  img = img.crop(self.crop_box)
    img = img.resize((cfg.width, cfg.height))
    img = np.array(img, dtype=np.float32)
    if(cfg.channel == 1):
      img = np.expand_dims(img, axis=2)
    return img
    
  def save_images(self, imgs, epoch=0):
    """Save images
   
    args:
      imgs: images in shape of [BatchSize, Weight, Height, Channel], must be normalized to [0,255]
      epoch: epoch number
    """
    imgs = imgs.astype('uint8')  # inverse_transform
    if(cfg.channel == 1):
      imgs = imgs[:,:,:,0]
    img_num = imgs.shape[0]
    test_size = self.test_list.shape[0]
    save_path = cfg.results + '/epoch'+str(epoch)
    if not os.path.exists(save_path):
      os.mkdir(save_path)
    for i in range(imgs.shape[0]):
      try:
        img_name = self.test_list[i + self.test_index - img_num].split('/')[-1]
      except:
        img_name = self.test_list[test_size + i + self.test_index - img_num].split('/')[-1]
      Image.fromarray(imgs[i]).save(os.path.join(save_path, img_name))
 
  
  
  
