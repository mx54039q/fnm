#coding: utf-8
import os
import tensorflow as tf
from config import cfg
#from LSGAN import LSGAN
from WGAN import WGAN
#from WGAN_GP import WGAN_GP

# Training Setting
test_num = 180 / cfg.batch_size

def arr2str(arr):
    arr = arr.reshape(cfg.batch_size,-1).mean(axis=1)
    s = ''
    for i in arr:
        s += ('%.2f,' % i)
    return s
    
def main(_):
    net = WGAN() #
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
        _,summary_str = sess.run([net.train_dis,net.train_summary], {net.is_train:True})
        writer.add_summary(summary_str)

        #fd_results = open('fd_results.txt', 'w')   
        
        # 1. Warm-Up: GAN loss not included
        if not cfg.is_finetune:
            for step in range(3000):
                _,fl,fl2,gs,gen = sess.run([net.train_wu,net.feature_loss,net.front_loss,
                                           net.global_step,net.texture_224], {net.is_train:True})
                print('Warm-Up-%d:, Fea Loss:%.3f, Front Loss:%.3f, gs:%d' % (step,fl,fl2,gs))
                
                if step % cfg.test_sum_freq == 0:
                    net.data_feed.save_train(gen)
                    for i in range(test_num):
                        te_profile, te_front = net.data_feed.get_test_batch(cfg.batch_size)
                        images = sess.run(net.texture_224,{net.profile:te_profile, net.front:te_front, net.is_train:True})
                        net.data_feed.save_images(images, 0)
                    print('Testing')
        saver.save(sess, cfg.logdir + '-wu')#
                    
        # 2. Train by minibatch and critic equals to 5
        for epoch in range(cfg.epoch):
            for step in range(num_batch):                
                # Discriminator Part
                if(step < 25 and epoch == 0 and not cfg.is_finetune):
                    critic = 25
                else:
                    critic = cfg.critic
                for i in range(critic):
                    _,dl,real,fake = sess.run([net.train_dis,net.d_loss,net.dr_224,net.df_224],
                                                   {net.is_train:True}) # net.clip_D,
                #fd_results.write('real:'+arr2str(real)+' fake:'+arr2str(fake)+'\n')
                #fd_results.flush()
                
                # Generative Part
                _,fl,fl2,gl,gs,gen = sess.run([net.train_gen,net.feature_loss,net.front_loss,net.g_loss,\
                                              net.global_step,net.texture_224], {net.is_train:True})
                print('Epoch-Step: %d-%d, Fea Loss:%.3f, Front Loss:%.3f, D Loss:%.3f, G Loss:%.3f, gs:%d' % 
                    (epoch, step, fl, fl2, dl, gl, gs))
                
                if step % cfg.test_sum_freq == 0:
                    net.data_feed.save_train(gen)
                    fl, fl2,dl, gl = 0., 0., 0., 0.
                    for i in range(test_num):
                        te_profile, te_front = net.data_feed.get_test_batch(cfg.batch_size)
                        dl_, gl_, fl_,fl2_, images,real,fake = sess.run([net.d_loss,net.g_loss,\
                            net.feature_loss,net.front_loss, net.texture_224, net.dr_224,net.df_224],
                            {net.profile:te_profile, net.front:te_front, net.is_train:True})
                        net.data_feed.save_images(images, epoch)
                        dl += dl_; gl += gl_; fl += fl_; fl2 += fl2_
                    print('Testing: Fea Loss:%.3f, Front Loss:%.3f, D Loss:%.3f, G Loss:%.3f' % 
                         (fl/test_num, fl2/test_num, dl/test_num, gl/test_num))
                if step % cfg.save_freq == cfg.save_freq - 1:
                    saver.save(sess, cfg.logdir + '-%02d' % (epoch))#
            
        #fd_results.close()
        
if __name__ == "__main__":
    tf.app.run()
