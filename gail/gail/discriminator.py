
import numpy as np
import tensorflow as tf
# from utils import get_shape, batch_normalization

def lkrelu(x, slope = 0.05):
    #slope斜率
    return tf.maximum(slope * x, x)


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            # np.prod()函数用来计算所有元素的乘积
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(loc=0.0, scale=1.0, size=flat_shape)
        # loc：float 此概率分布的均值（对应着整个分布的中心centre）
        # scale：float 此概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高）
        # size：int or tuple of ints 输出的shape，默认为None，只输出一个值
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

class Discriminator(object):
    def __init__(self, arch_params, stddev = 0.02):
        #stddev: The standard deviation标准差
        self.stddev = stddev
        self.arch_params = arch_params

    def __call__(self, state, action, is_training, reuse = None):  # 将Discriminator变为可调用对象

        # t = tf.truncated_normal_initializer(stddev=0.1, seed=1)
        # 从截断的正态分布中输出随机值。生成的值服从具有指定平均值和标准偏差的正态分布，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择。
        with tf.variable_scope('discriminator', initializer = tf.truncated_normal_initializer(stddev = self.stddev), reuse = reuse):

            with tf.variable_scope('layer_001'):
                '''将state和action连接'''
                pairs = tf.concat([state, action], axis=1)
                # a.get_shape()中a的数据类型只能是tensor,且返回的是一个元组（tuple） x_shape=x.get_shape().as_list()
                nin = pairs.get_shape()[1].value
                v = self.arch_params['layer_001']
                # 在强化学习里面一般都在使用正交初始化;在openai.baselines.ppo2里面是利用SVD分解来做的
                w_1 = tf.get_variable("w", [nin, v['nh']], initializer=ortho_init(v['init_scale']))
                b_1 = tf.get_variable("b", [v['nh']], initializer=tf.constant_initializer(v['init_bias']))
                a_1 = tf.matmul(pairs, w_1) + b_1
                # a_1 = tf.layers.batch_normalization(a_1, training=is_training)
                a_1 = lkrelu(a_1)

            with tf.variable_scope('layer_002'):
                nin = a_1.get_shape()[1].value
                v = self.arch_params['layer_002']
                # 在强化学习里面一般都在使用正交初始化;在openai.baselines.ppo2里面是利用SVD分解来做的
                w_2 = tf.get_variable("w", [nin, v['nh']], initializer=ortho_init(v['init_scale']))
                b_2 = tf.get_variable("b", [v['nh']], initializer=tf.constant_initializer(v['init_bias']))
                a_2 = tf.matmul(a_1, w_2) + b_2
                # a_2 = tf.layers.batch_normalization(a_2, training=is_training)
                a_2 = lkrelu(a_2)

            with tf.variable_scope('layer_003'):
                nin = a_2.get_shape()[1].value
                v = self.arch_params['layer_003']
                #在强化学习里面一般都在使用正交初始化;在openai.baselines.ppo2里面是利用SVD分解来做的
                w_3 = tf.get_variable("w", [nin, v['nh']], initializer=ortho_init(v['init_scale']))
                b_3 = tf.get_variable("b", [v['nh']], initializer=tf.constant_initializer(v['init_bias']))
                a_3 = tf.matmul(a_2, w_3) + b_3
                # a_3 = tf.layers.batch_normalization(a_3, training=is_training)
                a_3 = lkrelu(a_3)

            with tf.variable_scope('layer_004'):
                nin = a_3.get_shape()[1].value
                v = self.arch_params['layer_004']
                # 在强化学习里面一般都在使用正交初始化;在openai.baselines.ppo2里面是利用SVD分解来做的
                w_4 = tf.get_variable("w", [nin, v['nh']], initializer=ortho_init(v['init_scale']))
                b_4 = tf.get_variable("b", [v['nh']], initializer=tf.constant_initializer(v['init_bias']))
                a_4 = tf.matmul(a_3, w_4) + b_4
                # a_4 = tf.layers.batch_normalization(a_4, training=is_training)
                a_4 = tf.sigmoid(a_4)

            return a_4

# print(np.random.uniform())