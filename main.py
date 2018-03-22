#coding: utf-8
import os
import tensorflow as tf
from config import cfg
#from LSGAN import LSGAN
from WGAN_GP import WGAN_GP
#from WGAN_GP2 import WGAN_GP

# Training Setting
test_num = 1054 / cfg.batch_size

def arr2str(arr):
    arr = arr.reshape(cfg.batch_size,-1).mean(axis=1)
    s = ''
    for i in arr:
        s += ('%.2f,' % i)
    return s
    
def main(_):
    # Choose GPU 
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    net = WGAN_GP() #
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
        
        # 1. Warm-Up: GAN loss not included
        if 0:
        #not cfg.is_finetune:
            for step in range(1000):
                _,fl,fl2,gs,gen = sess.run([net.train_wu,net.feature_loss,net.front_loss,
                                           net.global_step,net.gen_p], {net.is_train:True})
                print('Warm-Up-%d:, Fea Loss:%.3f, Front Loss:%.3f, gs:%d' % (step,fl,fl2,gs))
                
                if step % cfg.test_sum_freq == 0:
                    net.data_feed.save_train(gen)
                    for i in range(test_num):
                        te_profile, te_front = net.data_feed.get_test_batch(cfg.batch_size)
                        images = sess.run(net.gen_p,{net.profile:te_profile, net.front:te_front, net.is_train:True})
                        net.data_feed.save_images(images, 0)
                    print('Testing')
            saver.save(sess, cfg.logdir + '-wu')#
                    
        # 2. Train by minibatch and critic
        for epoch in range(cfg.epoch):
            for step in range(num_batch):                
                # Discriminator Part
                if(step < 25 and epoch == 0 and not cfg.is_finetune):
                    critic = 25
                else:
                    critic = cfg.critic
                for i in range(critic - 1):
                    _ = sess.run(net.train_dis, {net.is_train:True}) # net.clip_D,
                
                # Generative Part
                _,_,fl,gl,dl,gen,g1,g2,g4,summary = sess.run([net.train_gen,net.train_dis,net.feature_loss,net.g_loss,
                                                       net.d_loss,net.gen_p,net.grad1,net.grad2,net.grad4,net.train_summary],
                                                       {net.is_train:True})
                writer.add_summary(summary, epoch*num_batch + step)
                print('%d-%d, Fea Loss:%.2f, D Loss:%4.1f, G Loss:%4.1f, g1/2/4:%.3f/%.3f/%.3f' % 
                    (epoch, step, fl, dl, gl, g1*100,g2,g4))
                
                if step % cfg.test_sum_freq == 0:
                    net.data_feed.save_train(gen)
                    fl, dl, gl = 0., 0., 0.
                    for i in range(test_num):
                        te_profile, te_front = net.data_feed.get_test_batch(cfg.batch_size)
                        dl_, gl_, fl_, images = sess.run([net.d_loss,net.g_loss,\
                                                          net.feature_loss, net.gen_p],
                                                          {net.profile:te_profile, net.front:te_front, net.is_train:False})
                        net.data_feed.save_images(images, epoch)
                        dl += dl_; gl += gl_; fl += fl_
                    print('Testing: Fea Loss:%.1f, D Loss:%.1f, G Loss:%.1f' % 
                         (fl/test_num, dl/test_num, gl/test_num))
                if step % cfg.save_freq == cfg.save_freq - 1:
                    print("Saving Model....")
                    saver.save(sess, cfg.logdir + '-%02d' % (epoch))#
            
        #fd_results.close()
        
if __name__ == "__main__":
    tf.app.run()
