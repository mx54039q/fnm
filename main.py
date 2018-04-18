#coding: utf-8
import os
import tensorflow as tf
from config import cfg
from WGAN_GP import WGAN_GP
#from WGAN_GP2 import WGAN_GP

# Training Setting
test_num = 800 / cfg.batch_size
    
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
        saver = tf.train.Saver(max_to_keep=0) #
        if cfg.is_finetune:
            saver.restore(sess, cfg.model_path)
            print('Load Model Successfully!')
        
        num_batch = int(cfg.dataset_size / cfg.batch_size)
        writer = tf.summary.FileWriter(cfg.summary_dir, sess.graph)
                    
        # 2. Train by minibatch and critic
        for epoch in range(cfg.epoch):
            for step in range(num_batch):                
                # Discriminator Part
                if(step < 25 and epoch == 0 and not cfg.is_finetune):
                    critic = 25
                else:
                    critic = cfg.critic
                for i in range(critic):
                    _ = sess.run(net.train_dis, {net.is_train:True}) # net.clip_D,
                
                # Generative Part
                _,fl,gl,dl,gen,g1,g2,g4,summary = sess.run([net.train_gen,net.feature_loss,net.g_loss,
                                                       net.d_loss,net.gen_p,net.grad1,net.grad2,net.grad4,net.train_summary],
                                                       {net.is_train:True})
                writer.add_summary(summary, epoch*num_batch + step)
                print('%d-%d, Fea Loss:%.2f, D Loss:%4.1f, G Loss:%4.1f, g1/2/4:%.3f/%.3f/%.3f' % 
                    (epoch, step, fl, dl, gl, g1,g2,g4))
                
                if step % cfg.test_sum_freq == 0:
                    net.data_feed.save_train(gen)
                    fl, dl, gl = 0., 0., 0.
                    for i in range(test_num):
                        te_profile, te_front = net.data_feed.get_test_batch(cfg.batch_size)
                        dl_, gl_, fl_, images = sess.run([net.d_loss,net.g_loss,\
                                                          net.feature_loss, net.gen_p],
                                                          {net.profile:te_profile, net.front:te_front, net.is_train:False}) #
                        net.data_feed.save_images(images, epoch)
                        dl += dl_; gl += gl_; fl += fl_
                    print('Testing: Fea Loss:%.1f, D Loss:%.1f, G Loss:%.1f' % 
                         (fl/test_num, dl/test_num, gl/test_num))
                if(step != 0 and step % cfg.save_freq == 0):
                    print("Saving Model....")
                    saver.save(sess, cfg.logdir + '-%1d-%d' % (epoch,step)) #
            
        #fd_results.close()
        
if __name__ == "__main__":
    tf.app.run()
