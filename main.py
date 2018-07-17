#coding: utf-8
import os
import tensorflow as tf
from config import cfg
from WGAN_GP import WGAN_GP

# Training Setting
test_num = 800 / cfg.batch_size
    
def main(_):
    # Environment Setting
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    if not os.path.exists(cfg.results):
        os.mkdir(cfg.results)
    
    # Construct Networks
    # Change this line if 'LSGAN' or 'WGAN'
    net = WGAN_GP()
    
    # Train and Test
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=net.graph) as sess:
        sess.run(tf.global_variables_initializer())
        
        # Start Thread
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        saver = tf.train.Saver(max_to_keep=0) #
        if cfg.is_finetune:
            saver.restore(sess, cfg.model_path)
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
                    # add 'net.clip_D' into ops if 'LSGAN' or 'WGAN'
                    _ = sess.run(net.train_dis, {net.is_train:True}) # net.clip_D
                
                # Generative Part
                _,fl,gl,dl,gen,summary = sess.run([net.train_gen,net.feature_loss,net.g_loss,
                                                  net.d_loss,net.gen_p,net.train_summary],
                                                 {net.is_train:True})
                writer.add_summary(summary, epoch*num_batch + step)
                print('%d-%d, Fea Loss:%.2f, D Loss:%4.1f, G Loss:%4.1f' %  #g1/2/3:%.5f/%.5f/%.5f 
                     (epoch, step, fl, dl, gl)) #g1*cfg.lambda_fea,g2,g4
                
                # Test Part
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
                    print('Testing: Fea Loss:%.1f, D Loss:%.1f, G Loss:%.1f' % (fl/test_num, dl/test_num, gl/test_num))
                    
                # Save Model
                if(step != 0 and step % cfg.save_freq == 0):
                    print("Saving Model....")
                    saver.save(sess, cfg.logdir + '-%02d' % (epoch)) #
        
        # Close Threads
        coord.request_stop()
        coord.join(threads)
        
if __name__ == "__main__":
    tf.app.run()
