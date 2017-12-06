#coding: utf-8
import os
import tensorflow as tf
from config import cfg
from Net import Net

"""
"""

# Training Setting
test_num = 90 / cfg.batch_size

def main(_):
    net = Net()
    # Net.fc7_encoder: feature of the input images
    # Net.texture: frontal result 
    # Net.recon_feature: feature of the texture
    # Net.recon_feature_gt: feature of the ground true front
    
    if not os.path.exists(cfg.results):
        os.mkdir(cfg.results)
        
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=net.graph) as sess:
        sess.run(tf.global_variables_initializer())
        threads = tf.train.start_queue_runners(sess=sess)
        saver = tf.train.Saver(max_to_keep=1)
        if cfg.is_finetune:
            saver.restore(sess, cfg.model_path)
            print('Load Model Successfully!')
        
        num_batch = int(cfg.dataset_size / cfg.batch_size)
        
        # 1. Only texture loss and feature loss
        if not cfg.is_finetune:
            for step in range(num_batch):
                global_step = sess.run(net.global_step)
                _, lf1, lf2, lr = sess.run([net.train_texture,net.front_loss,net.feature_loss,net.lr],
                    {net.is_train:True})
                print('Pretrain-Step: %d, Front Loss:%.4f, Feature Loss:%.4f, lr:%.5f' % 
                    (step, lf1, lf2, lr))

                if step % cfg.test_sum_freq == 0:
                    fl1, fl2 = 0, 0
                    for i in range(test_num):
                        te_profile, te_front = net.data_feed.get_test_batch(cfg.batch_size)
                        fl1_, fl2_, images = sess.run([net.front_loss,net.feature_loss,net.texture],
                            {net.profile:te_profile, net.front:te_front, net.is_train:False})
                        net.data_feed.save_images(images, 0)
                        fl1 += fl1_; fl2 += fl2_
                    print('Testing: Front Loss:%.4f, Feature Loss:%.4f' % 
                        (fl1/test_num, fl2/test_num))
                    
                if step == num_batch - 1:
                    saver.save(sess, cfg.logdir + '-%04d-%02d' % (0, global_step))#
                
        # 2. Join GAN loss
        for epoch in range(cfg.epoch):
            for step in range(num_batch):
                global_step = sess.run(net.global_step)
                
                # Discriminator Part
                _, dl, fl, lr = sess.run([net.train_dis,net.d_loss,net.front_loss,net.lr],
                    {net.is_train:True})
                # Generative Part Twice
                _, gl = sess.run([net.train_gen, net.g_loss],
                    {net.is_train:True})
                _, gl = sess.run([net.train_gen, net.g_loss],
                    {net.is_train:True})
                print('Epoch-Step: %d-%d, Front Loss:%.4f, D Loss:%.4f, G Loss:%.4f, lr:%.5f' % 
                    (epoch, step, fl, dl, gl, lr))
                
                if step % cfg.test_sum_freq == 0:
                    fl, dl, gl = 0, 0, 0
                    for i in range(test_num):
                        te_profile, te_front = net.data_feed.get_test_batch(cfg.batch_size)
                        dl_, gl_, fl_, images = sess.run([net.d_loss,net.g_loss,net.front_loss,net.texture],
                            {net.profile:te_profile, net.front:te_front, net.is_train:False})
                        net.data_feed.save_images(images, epoch)
                        dl += dl_; gl += gl_; fl += fl_
                    print('Testing: Front Loss:%.4f, D Loss:%.4f, G Loss:%.4f' % 
                        (fl/test_num, dl/test_num, gl/test_num))

                if step == num_batch - 1:
                    saver.save(sess, cfg.logdir + '-%04d-%02d' % (epoch, global_step))#

if __name__ == "__main__":
    tf.app.run()
