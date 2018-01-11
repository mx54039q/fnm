#coding: utf-8
import os
import tensorflow as tf
from config import cfg
from Net2 import Net

"""
"""

# Training Setting
test_num = 90 / cfg.batch_size


def arr2str(arr):
    arr = arr.reshape(cfg.batch_size,-1).mean(axis=1)
    s = ''
    for i in arr:
        s += ('%.2f,' % i)
    return s
    
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
        saver = tf.train.Saver()
        if cfg.is_finetune:
            saver.restore(sess, cfg.model_path)
            print('Load Model Successfully!')
        
        
        num_batch = int(cfg.dataset_size / cfg.batch_size)
        
        writer = tf.summary.FileWriter(cfg.summary_dir, sess.graph)
        _, summary_str = sess.run([net.train_texture, net.train_summary], {net.is_train:True})
        writer.add_summary(summary_str)
        
        # 1. Only texture loss and feature loss
        if 0:
        #not cfg.is_finetune:
            for step in range(num_batch/2):
                global_step = sess.run(net.global_step)
                _, lf1, lf2, lr = sess.run([net.train_texture,net.front_loss,net.feature_loss,net.lr],
                    {net.is_train:True})
                print('Warm Up: %d, Front Loss:%.4f, Feature Loss:%.4f, gs:%d' % 
                    (step, lf1, lf2, global_step))

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
            saver.save(sess, cfg.logdir + '-pretrain')#
        
        # 2_ : Warm Up( Only GAN loss)
        fd_results = open('fd_results.txt', 'w')
        test_results = open('test_results.txt', 'w')    
         
        # 2. Join GAN loss
        for epoch in range(cfg.epoch):
            for step in range(num_batch):                
                # Discriminator Part
                _, dl, fl, gen,real,fake = sess.run([net.train_dis,net.d_loss,net.front_loss,net.texture_224,net.dr_224,net.df_224],
                    {net.is_train:True})
                fd_results.write('real:'+arr2str(real)+' fake:'+arr2str(fake)+'\n')
                fd_results.flush()
                
                # Generative Part 
                sess.run([net.train_gen], {net.is_train:True})
                _, fl2, gl, global_step = sess.run([net.train_gen,net.feature_loss, net.g_loss, net.global_step],
                    {net.is_train:True})
                print('Epoch-Step: %d-%d, Front Loss:%.3f, Fea Loss:%.3f, D Loss:%.3f, G Loss:%.3f, gs:%d' % 
                    (epoch, step, fl, fl2, dl, gl, global_step))
                
                if step % cfg.test_sum_freq == 0:
                    net.data_feed.save_train(gen)
                    fl, fl2, dl, gl = 0, 0, 0, 0
                    for i in range(test_num):
                        te_profile, te_front = net.data_feed.get_test_batch(cfg.batch_size)
                        dl_, gl_, fl_, fl2_, images,real,fake = sess.run([net.d_loss,net.g_loss,net.front_loss,\
                            net.feature_loss, net.texture_224, net.dr_224,net.df_224],
                            {net.profile:te_profile, net.front:te_front, net.is_train:True})
                        test_results.write('real:'+arr2str(real)+' fake:'+arr2str(fake)+'\n')
                        test_results.flush()
                        net.data_feed.save_images(images, epoch)
                        dl += dl_; gl += gl_; fl += fl_; fl2 += fl2_
                    print('Testing: Front Loss:%.4f, Fea Loss:%.3f, D Loss:%.4f, G Loss:%.4f' % 
                        (fl/test_num, fl2/test_num, dl/test_num, gl/test_num))

            saver.save(sess, cfg.logdir + '-%02d' % (epoch))#
            
        fd_results.close()
        test_results.close()
        
if __name__ == "__main__":
    tf.app.run()
