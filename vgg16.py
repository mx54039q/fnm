import inspect
import os

import numpy as np
import tensorflow as tf
import time

VGG_MEAN = [129.1836, 104.7624, 93.5940] # for channel BGR
#VGG_MEAN = [103.939, 116.779, 123.68] 

class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg_face.npy")
            vgg16_npy_path = path

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")
    
    def build(self):
        """
        load parameters from npy to build the VGG
        """
        with tf.variable_scope('vgg'):
            with tf.variable_scope('conv1_1'):
                self.conv1_1_weights, self.conv1_1_bias = self.get_filter_bias('conv1_1')
            with tf.variable_scope('conv1_2'):
                self.conv1_2_weights, self.conv1_2_bias = self.get_filter_bias('conv1_2')
                    
            with tf.variable_scope('conv2_1'):
                self.conv2_1_weights, self.conv2_1_bias = self.get_filter_bias('conv2_1')
            with tf.variable_scope('conv2_2'):
                self.conv2_2_weights, self.conv2_2_bias = self.get_filter_bias('conv2_2')

            with tf.variable_scope('conv3_1'):
                self.conv3_1_weights, self.conv3_1_bias = self.get_filter_bias('conv3_1')
            with tf.variable_scope('conv3_2'):
                self.conv3_2_weights, self.conv3_2_bias = self.get_filter_bias('conv3_2')
            with tf.variable_scope('conv3_3'):
                self.conv3_3_weights, self.conv3_3_bias = self.get_filter_bias('conv3_3')

            with tf.variable_scope('conv4_1'):
                self.conv4_1_weights, self.conv4_1_bias = self.get_filter_bias('conv4_1')
            with tf.variable_scope('conv4_2'):
                self.conv4_2_weights, self.conv4_2_bias = self.get_filter_bias('conv4_2')
            with tf.variable_scope('conv4_3'):
                self.conv4_3_weights, self.conv4_3_bias = self.get_filter_bias('conv4_3')

            with tf.variable_scope('conv5_1'):
                self.conv5_1_weights, self.conv5_1_bias = self.get_filter_bias('conv5_1')
            with tf.variable_scope('conv5_2'):
                self.conv5_2_weights, self.conv5_2_bias = self.get_filter_bias('conv5_2')
            with tf.variable_scope('conv5_3'):
                self.conv5_3_weights, self.conv5_3_bias = self.get_filter_bias('conv5_3')

            with tf.variable_scope('fc6'):
                self.fc6_weights, self.fc6_bias = self.get_filter_bias('fc6')
            with tf.variable_scope('fc7'):
                self.fc7_weights, self.fc7_bias = self.get_filter_bias('fc7')
            with tf.variable_scope('fc8'):
                self.fc8_weights, self.fc8_bias = self.get_filter_bias('fc8')
        
        # Clear the model dict
        self.data_dict = None 
        print("build model finished.")
        
    def forward(self, rgb, reuse=False):
        """
        forword pass 
        args:
            rgb: rgb image tensors with shape(batch, height, width, 3), values range[0,255]
            reuse: whether to reuse the variables. (default False)
        return:
            pool5: output of pool5 layers
            relu7: output of relu7 layers
        """
        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        
        with tf.name_scope('conv1_1'):
            conv1_1 = tf.nn.conv2d(bgr, self.conv1_1_weights, [1, 1, 1, 1], padding='SAME')
            conv1_1 = tf.nn.relu(tf.nn.bias_add(conv1_1, self.conv1_1_bias))
        with tf.name_scope('conv1_2'):
            conv1_2 = tf.nn.conv2d(conv1_1, self.conv1_2_weights, [1, 1, 1, 1], padding='SAME')
            conv1_2 = tf.nn.relu(tf.nn.bias_add(conv1_2, self.conv1_2_bias))
        pool1 = self.max_pool(conv1_2, 'pool1')
        # output shape: [112, 112, 64]

        with tf.name_scope('conv2_1'):
            conv2_1 = tf.nn.conv2d(pool1, self.conv2_1_weights, [1, 1, 1, 1], padding='SAME')
            conv2_1 = tf.nn.relu(tf.nn.bias_add(conv2_1, self.conv2_1_bias))
        with tf.name_scope('conv2_2'):
            conv2_2 = tf.nn.conv2d(conv2_1, self.conv2_2_weights, [1, 1, 1, 1], padding='SAME')
            conv2_2 = tf.nn.relu(tf.nn.bias_add(conv2_2, self.conv2_2_bias))
        pool2 = self.max_pool(conv2_2, 'pool2')
        # output shape: [56, 56, 128]

        with tf.name_scope('conv3_1'):
            conv3_1 = tf.nn.conv2d(pool2, self.conv3_1_weights, [1, 1, 1, 1], padding='SAME')
            conv3_1 = tf.nn.relu(tf.nn.bias_add(conv3_1, self.conv3_1_bias))
        with tf.name_scope('conv3_2'):
            conv3_2 = tf.nn.conv2d(conv3_1, self.conv3_2_weights, [1, 1, 1, 1], padding='SAME')
            conv3_2 = tf.nn.relu(tf.nn.bias_add(conv3_2, self.conv3_2_bias))
        with tf.name_scope('conv3_3'):
            conv3_3 = tf.nn.conv2d(conv3_2, self.conv3_3_weights, [1, 1, 1, 1], padding='SAME')
            conv3_3 = tf.nn.relu(tf.nn.bias_add(conv3_3, self.conv3_3_bias))
        pool3 = self.max_pool(conv3_3, 'pool3')
        # output shape: [28, 28, 256]

        with tf.name_scope('conv4_1'):
            conv4_1 = tf.nn.conv2d(pool3, self.conv4_1_weights, [1, 1, 1, 1], padding='SAME')
            conv4_1 = tf.nn.relu(tf.nn.bias_add(conv4_1, self.conv4_1_bias))
        with tf.name_scope('conv4_2'):
            conv4_2 = tf.nn.conv2d(conv4_1, self.conv4_2_weights, [1, 1, 1, 1], padding='SAME')
            conv4_2 = tf.nn.relu(tf.nn.bias_add(conv4_2, self.conv4_2_bias))
        with tf.name_scope('conv4_3'):
            conv4_3 = tf.nn.conv2d(conv4_2, self.conv4_3_weights, [1, 1, 1, 1], padding='SAME')
            conv4_3 = tf.nn.relu(tf.nn.bias_add(conv4_3, self.conv4_3_bias))
        pool4 = self.max_pool(conv4_3, 'pool4')
        # output shape: [14, 14, 512]
        
        with tf.name_scope('conv5_1'):
            conv5_1 = tf.nn.conv2d(pool4, self.conv5_1_weights, [1, 1, 1, 1], padding='SAME')
            conv5_1 = tf.nn.relu(tf.nn.bias_add(conv5_1, self.conv5_1_bias))
        with tf.name_scope('conv5_2'):
            conv5_2 = tf.nn.conv2d(conv5_1, self.conv5_2_weights, [1, 1, 1, 1], padding='SAME')
            conv5_2 = tf.nn.relu(tf.nn.bias_add(conv5_2, self.conv5_2_bias))
        with tf.name_scope('conv5_3'):
            conv5_3 = tf.nn.conv2d(conv5_2, self.conv5_3_weights, [1, 1, 1, 1], padding='SAME')
            conv5_3 = tf.nn.relu(tf.nn.bias_add(conv5_3, self.conv5_3_bias))
        pool5 = self.max_pool(conv5_3, 'pool5')
        # output shape: [7, 7, 512]

        with tf.name_scope('fc6'):
            shape = pool5.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(pool5, [-1, dim])
            fc6 = tf.nn.bias_add(tf.matmul(x, self.fc6_weights), self.fc6_bias)
        assert fc6.get_shape().as_list()[1:] == [4096]
        relu6 = tf.nn.relu(fc6)
        # output shape: [4096]
        
        with tf.name_scope('fc7'):
            fc7 = tf.nn.bias_add(tf.matmul(relu6, self.fc7_weights), self.fc7_bias)
        relu7 = tf.nn.relu(fc7)
        # output shape: [4096]
        
        with tf.name_scope('fc8'):
            fc8 = tf.nn.bias_add(tf.matmul(relu7, self.fc8_weights), self.fc8_bias)
        
        return pool2, pool5, relu7 #relu7


    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_filter_bias(self, name):
        return tf.Variable(self.data_dict[name]['weights'], name="weights"), \
                   tf.constant(self.data_dict[name]['biases'], name="biases")
    
    def get_filter(self, name):
        return tf.Variable(self.data_dict[name]['weights'], name="filter")

    def get_bias(self, name):
        return tf.Variable(self.data_dict[name]['biases'], name="biases")

