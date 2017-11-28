#coding: utf-8
import os
import tensorflow as tf
from tqdm import tqdm

from config import cfg # setting / configuration / hyper parameters
from utils import loadData # get data
from Net import Net # Class Net

"""
"""

def main(_):
    net = Net(is_training=cfg.is_training) # construct Capsule Network
    
    # Net.fc7_encoder: feature of the input images
    # Net.texture: frontal result 
    # Net.recon_feature: feature of the texture
    # Net.recon_feature_gt: feature of the ground true front
    
    data_feed = loadData(batch_size=cfg.batch_size, train_shuffle=False)
    tf.logging.info('Graph loaded')

    path = cfg.results + '/accuracy.csv'
    if not os.path.exists(cfg.results):
        os.mkdir(cfg.results)
    elif os.path.exists(path):
        os.remove(path)

    fd_results = open(path, 'w')
    fd_results.write('step,front_loss,feature_loss\n')
    config = tf.ConfigProto()
    
    #config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=net.graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1)
        if(cfg.is_fineture):
            saver.restore(sess, cfg.model_path)
        train_writer = tf.summary.FileWriter('logdir', sess.graph)
        
        num_batch = int(cfg.dataset_size / cfg.batch_size)
        #num_test_batch = 20860 // cfg.batch_size
        for epoch in range(cfg.epoch):
            for step in range(num_batch): #tqdm(range(num_batch)):
                tr_profile, tr_front = data_feed.get_train_batch() if cfg.use_profile
                    else data_feed.get_train_batch_feature()
                global_step = sess.run(net.global_step)
                _, lf1, lf2, lr = sess.run([net.train_op,net.front_loss,net.feature_loss,net.lr],
                    {net.profile:tr_profile, net.front:tr_front})
                print('Epoch-Step: %d-%d, Front Loss:%.4f, Feature Loss:%.4f, lr:%.5f' % 
                    (epoch, step, lf1, lf2, lr))
                
                if step % cfg.train_sum_freq == 0:
                    _, summary_str = sess.run([net.train_op, net.train_summary],
                        {net.profile:tr_profile, net.front:tr_front})
                    train_writer.add_summary(summary_str, global_step)

                if (global_step + 1) % cfg.test_sum_freq == 0:
                    #data_feed.test_index = 0
                    loss1, loss2 = 0., 0.
                    
                    #for i in range(num_test_batch):
                    te_profile, te_front = data_feed.get_test_batch(60) if cfg.use_profile
                        else data_feed.get_test_batch_feature(60)
                    l1, l2, images = sess.run([net.front_loss,net.feature_loss,net.texture],
                        {net.profile:te_profile, net.front:te_front})
                    
                    loss1 += l1
                    loss2 += l2
                    #images = sess.run(net.texture, {net.profile:te_profile, net.front:te_front})
                    data_feed.save_images(images, epoch)
                    #loss1 = loss1 / num_test_batch
                    #loss2 = loss2 / num_test_batch
                    fd_results.write(str(global_step + 1) + ',' + str(loss1) + ','+str(loss2) + '\n')
                    fd_results.flush()

                if step == num_batch - 1:
                    saver.save(sess, cfg.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))

    fd_results.close()
    tf.logging.info('Training done')


if __name__ == "__main__":
    tf.app.run()
