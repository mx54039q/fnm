# coding: utf-8
# --------------------------------------------------------
# FNM
# Written by Yichen Qian
# --------------------------------------------------------

import os
import tensorflow as tf
from config import cfg
from WGAN_GP import WGAN_GP
from utils import loadData
  
def main(_):
  # Environment Setting
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device_id
  
  if not os.path.exists(cfg.results):
    os.makedirs(cfg.results)
  if not os.path.exists(cfg.checkpoint):
    os.makedirs(cfg.checkpoint)
  if not os.path.exists(cfg.summary_dir):
    os.makedirs(cfg.summary_dir)
  
  # Construct Networks
  net = WGAN_GP()
  data_feed = loadData(batch_size=cfg.batch_size, train_shuffle=True) # False
  profile, front = data_feed.get_train()
  net.build_up(profile, front)
  
  # Train or Test
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config, graph=net.graph) as sess:
    sess.run(tf.global_variables_initializer())
    
    # Start Thread
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    saver = tf.train.Saver(max_to_keep=0)  #
    if cfg.is_finetune:
      saver.restore(sess, cfg.checkpoint_ft)
      print('Load Finetuned Model Successfully!')
      
    num_batch = int(cfg.dataset_size / cfg.batch_size)
    writer = tf.summary.FileWriter(cfg.summary_dir, sess.graph)
          
    # Train by minibatch and critic
    for epoch in range(cfg.epoch):
      for step in range(num_batch):        
        # Discriminator Part
        if(step < 25 and epoch == 0 and not cfg.is_finetune):
          critic = 25
        else:
          critic = cfg.critic
        for i in range(critic):
          _ = sess.run(net.train_dis, {net.is_train:True})
        
        # Generative Part
        #_,fl,gl,dl,gen,summary = sess.run([net.train_gen,net.feature_loss,net.g_loss,
        #                                   net.d_loss,net.gen_p,net.train_summary],
        #                                   {net.is_train:True})
        _,fl,gl,dl,gen,g1,g2,g4,summary = sess.run([net.train_gen, net.feature_loss,net.g_loss,
                            net.d_loss,net.gen_p,net.grad1,net.grad2,net.grad4,net.train_summary],
                           {net.is_train:True})
        
        #print('%d-%d, Fea Loss:%.2f, D Loss:%4.1f, G Loss:%4.1f,' % (epoch, step, fl, dl, gl))
        print('%d-%d, Fea Loss:%.2f, D Loss:%4.1f, G Loss:%4.1f, g1/2/4:%.5f/%.5f/%.5f ' %  #
           (epoch, step, fl, dl, gl, g1*cfg.lambda_fea,g2,g4))                 
        
        # Save Model and Summary and Test
        if(step % cfg.save_freq == 0):
          writer.add_summary(summary, epoch*num_batch + step)
          print("Saving Model....")
          saver.save(sess, os.path.join(cfg.checkpoint, 'ck-%02d' % (epoch))) #
          
          # test
          fl, dl, gl = 0., 0., 0.
          for i in range(50): # 25791 / 16
            te_profile, te_front = data_feed.get_test_batch(cfg.batch_size)
            dl_, gl_, fl_, images = sess.run([net.d_loss,net.g_loss, net.feature_loss, net.gen_p],
                              {profile:te_profile, front:te_front, net.is_train:False}) #
            data_feed.save_images(images, epoch)
            dl += dl_; gl += gl_; fl += fl_
          print('Testing: Fea Loss:%.1f, D Loss:%.1f, G Loss:%.1f' % (fl, dl, gl))
    
    # Close Threads
    coord.request_stop()
    coord.join(threads)
    
    
if __name__ == "__main__":
  tf.app.run()
