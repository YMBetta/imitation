########################################################################################################################
#@author Kai-Feng Zhang    email: zhangkf@lamda.nju.edu.cn or kfzhang.alex@gmail.com
#this file include some useful functions
########################################################################################################################

import numpy as np
import tensorflow as tf
import random
import scipy.signal

def get_shape(tensor): # static shape
    return tensor.get_shape().as_list()

def batch_normalization(*args, **kwargs):
    with tf.name_scope('bn'):
        bn = tf.layers.batch_normalization(*args, **kwargs)
    return bn