import tensorflow as tf
import copy
import numpy as np
from discriminator import Discriminator
import config
import gym
from gym import error, spaces

# x=tf.constant([[1,2,3],[4,5,6]])
# print(x.get_shape()[0].value)
# print(x.get_shape()[1].value)
#
# self.expert_state = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.env.observation_space.shape))
#
# self.obs = np.zeros((nenv,) + env.bservation_space.shape, dtype=model.train_model.X.dtype.name)
#
# self.discriminator_loss = - tf.reduce_mean(tf.log(self.discriminator_expert_output + configs.epsilon) + tf.log(
#             1 - self.discriminator_gen_output + configs.epsilon))
#
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
#
#
# with tf.Session() as sess:
#     output = sess.run(hello_constant)


'''tf.argmax'''


def test_argmax():
    A = [[1, 3, 4, 5, 6]]
    B = [[1, 3, 4], [2, 4, 1]]

    with tf.Session() as sess:
        print(sess.run(tf.argmax(A, 1)))
        print(sess.run(tf.argmax(B, 1)))

        print(sess.run(tf.argmax(A, 0)))
        print(sess.run(tf.argmax(B, 0)))


def test_001():
    def ortho_init(scale=1.0):
        def _ortho_init(shape, dtype, partition_info=None):
            # lasagne ortho init for tf
            # partition 分区
            shape = tuple(shape)
            if len(shape) == 2:
                flat_shape = shape
            elif len(shape) == 4:  # assumes NHWC
                # np.prod()函数用来计算所有元素的乘积
                flat_shape = (np.prod(shape[:-1]), shape[-1])
                # 输出前3个数的乘积，和第四个数
            else:
                raise NotImplementedError
            a = np.random.normal(loc=0.0, scale=1.0, size=flat_shape)
            # 所谓标准正态分布
            # loc：float 此概率分布的均值（对应着整个分布的中心centre）
            # scale：float 此概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高）
            # size：int or tuple of ints 输出的shape，默认为None，只输出一个值
            u, _, v = np.linalg.svd(a, full_matrices=False)
            q = u if u.shape == flat_shape else v  # pick the one with the correct shape
            q = q.reshape(shape)
            return (scale * q[:shape[0], :shape[1]]).astype(np.float32)

        return _ortho_init

    def lkrelu(x, slope=0.01):
        # slope斜率
        return tf.maximum(slope * x, x)

    def test_1_mlp(x):
        '''test 一层全连接'''
        with tf.variable_scope('layer_001'):
            # a.get_shape()中a的数据类型只能是tensor,且返回的是一个元组（tuple） x_shape=x.get_shape().as_list()
            nin = x.get_shape()[1].value

            # 在强化学习里面一般都在使用正交初始化;在openai.baselines.ppo2里面是利用SVD分解来做的
            w_1 = tf.get_variable(name="w", shape=[nin, 2], initializer=ortho_init(1.0))
            b_1 = tf.get_variable(name="b", shape=[2], initializer=tf.constant_initializer(0.0))
            a_1 = tf.matmul(x, w_1) + b_1
            a_1 = lkrelu(a_1)
            output = tf.add(a_1, 0, name='o')

        return output

    xx = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

    x = tf.placeholder(name='x', shape=[None, 3], dtype=tf.float32)
    a = test_1_mlp(x)

    saver = tf.train.Saver(max_to_keep=1)
    step = 0

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(a, feed_dict={x: xx.reshape([-1, 3])}))

        saver.save(sess, 'model/ckpt', global_step=step)


def test_restore():
    sess = tf.Session()
    saver = tf.train.import_meta_graph('model/ckpt-0.meta')
    model_file = tf.train.latest_checkpoint('model/')
    saver.restore(sess, model_file)
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('x:0')
    output = graph.get_tensor_by_name('layer_001/o:0')

    xx = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

    with sess.as_default():
        print(sess.run(output, feed_dict={x: xx.reshape([-1, 3])}))


# correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)
# acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def test_discriminator():
    from logger import MyLogger
    mylogger = MyLogger('./log')

    configs = config.configs['gail']
    print(tf.get_default_session())
    with tf.Session() as sess:
        d = Discriminator(arch_params=configs.discriminator_params, stddev=0.02)
        s1 = np.array([[1, 2, 3, 4],[1, 2, 3, 4]]).reshape([-1, 4])
        a1 = np.array([[1, -1],[1, -1]]).reshape([-1, 2])

        s2 = np.array([8, -6, 5, 7]).reshape([-1, 4])
        a2 = np.array([1, 1]).reshape([-1, 2])

        is_training = tf.placeholder(tf.bool)
        e_s = tf.placeholder(dtype=tf.float32, shape=list(s1.shape), name='e_s')
        e_a = tf.placeholder(dtype=tf.float32, shape=list(a1.shape), name='e_a')
        g_s = tf.placeholder(dtype=tf.float32, shape=list(s2.shape), name='g_s')
        g_a = tf.placeholder(dtype=tf.float32, shape=list(a2.shape), name='g_a')
        # print([None]+list(s1.shape))

        e_output = d(state=e_s, action=e_a, is_training=is_training)
        g_output = d(state=g_s, action=g_a, is_training=is_training, reuse=True)

        discriminator_loss = - tf.reduce_mean(
            tf.log(e_output + configs.epsilon) + tf.log(1 - g_output + configs.epsilon))

        # # tf.GraphKeys.UPDATE_OPS ,tf.GraphKeys.TRAINABLE_VARIABLES
        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        #     discriminator_train_step = tf.train.AdamOptimizer(configs.learning_rate, configs.beta1,
        #                                                       configs.beta2).minimize(discriminator_loss)

        discriminator_train_step = tf.train.AdamOptimizer(configs.learning_rate, configs.beta1,
                                                          configs.beta2).minimize(discriminator_loss,
                                                                                  var_list=tf.get_collection(
                                                                                      tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                      scope='discriminator'))

        tf.global_variables_initializer().run(session=sess)

        for i in range(100):
            eoutput, goutput, _, dloss = sess.run([e_output, g_output, discriminator_train_step, discriminator_loss],
                                                  feed_dict={e_s: s1,
                                                             e_a: a1,
                                                             g_s: s2,
                                                             g_a: a2,
                                                             is_training: False})
            s1 += 0
            a1 += 0
            mylogger.write_summary_scalar(iteration=i,tag ='dloss',value=dloss)
            mylogger.write_summary_scalar(iteration=i, tag='loss', value=dloss)


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    # 将指定的轴对换了 swap and then flatten axes 0 and 1
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def test_runner():
    # a = gym.spaces.Discrete(4)
    # b = gym.spaces.Box(high=np.array([1.,1.]),low=np.array([-1.,-1.]),dtype=np.float32)
    # print(type(obs))
    # print(b.shape)
    # print(b.sample())
    # print(obs.shape)
    # print(a.n)

    obs = np.array([[[1,2,3],[4,5,6]],[[-1,-2,-3],[-4,-5,-6]]])
    ac  = np.array([[[1,1]],[[1,1]]])
    print('obs shape:',obs.shape)

    #rtn = list(map(sf01, (obs, obs)))
    l = (*map(sf01,(obs)),)
    print(l)
    #print(*map(sf01,(obs,obs)),obs,obs)
    return(*map(sf01,(obs)),)

def test_book():

    a = tf.constant([1,2,3],name='a',dtype=tf.float32)
    '''查看一个计算所属的计算图，默认的计算图'''
    print(a.graph is tf.get_default_graph())

    graph =tf.Graph()
    with graph.as_default():
        a = tf.constant([1, 2, 3], name='a', dtype=tf.float32)

    graph2 = tf.Graph()
    with graph2.as_default():
        b = tf.constant([1, 2, 3], name='b', dtype=tf.float32)

    with tf.Session(graph=graph) as sess:
        #tf.global_variables_initializer().run(session=sess)
        sess.run(tf.global_variables_initializer())
        print(sess.run(a))

class Tuple(gym.Space):
    """
    A tuple (i.e., product) of simpler spaces
    Example usage:
    self.observation_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(3)))
    """
    def __init__(self, spaces):
        self.spaces = spaces
        gym.Space.__init__(self, None, None)

    def sample(self):
        return tuple([space.sample() for space in self.spaces])

    def contains(self, x):
        if isinstance(x, list):
            x = tuple(x)  # Promote list to tuple for contains check
        return isinstance(x, tuple) and len(x) == len(self.spaces) and all(
            space.contains(part) for (space,part) in zip(self.spaces,x))

    def __repr__(self):
        return "Tuple(" + ", ". join([str(s) for s in self.spaces]) + ")"

    def to_jsonable(self, sample_n):
        # serialize as list-repr of tuple of vectors
        return [space.to_jsonable([sample[i] for sample in sample_n]) \
                for i, space in enumerate(self.spaces)]

    def from_jsonable(self, sample_n):
        return [sample for sample in zip(*[space.from_jsonable(sample_n[i]) for i, space in enumerate(self.spaces)])]

    def __eq__(self, other):
        return self.spaces == other.spaces

def test_tensorboard():
    from logger import MyLogger
    mylogger = MyLogger('./log')
    input1 = tf.constant([1.0,2.0,3.0],name='input1')
    input2 = tf.Variable(tf.random_uniform([3]),name='inout2')
    output = tf.add_n([input1,input2],name='add')

if __name__ == '__main__':
    ''''''
    #test_tensorboard()
    #test_discriminator()
    observation_space = spaces.Box(low=np.zeros([1737]) - 1, high=np.zeros([1737]) + 1, dtype=np.float32)
    ob_shape = (100,) + observation_space.shape
    print(ob_shape)





    # test_argmax()
    # test_001()
    # test_restore()
    #test_discriminator()
    #
    # #print(1.0-True)
    # '''test runner'''
    # l = test_runner()
    # print(type(l))
    #

    # test_book()
    # action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([+1, +1]), dtype=np.float32)
    # c = spaces.Tuple([spaces.Box(low=np.zeros([1737]) - 1, high=np.zeros([1737]) + 1, dtype=np.float32)])
    # b = spaces.Discrete(5)
    #
    # print(c.spaces)




'''sess'''
# #build graph
# a=tf.constant(2.)
# b=tf.constant(5.)
# c=a*b
#
# #construct session
# sess=tf.Session()
#
# with sess.as_default():
#     print(sess.run(c))


'''Graph'''
# c=tf.constant(value=1)
# #print(assert c.graph is tf.get_default_graph())
# print(c.graph)
#
# g=tf.Graph()
# with g.as_default():
#     d=tf.constant(value=2)
#     print(d.graph)
#     #print(g)
#
# g2=tf.Graph()
# print("g2:",g2)
# g2.as_default()
# e=tf.constant(value=15)
# print(e.graph)


'''swapaxes'''
# def sf01(arr):
#     """
#     swap and then flatten axes 0 and 1
#     """
#     s = arr.shape
#     print(s[0],s[1],s[2])
#     return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
# mb_obs = np.asarray([[1,2,3],[4,5,6]])
# print(*map(sf01, (mb_obs)),)
# print(mb_obs)
# print(mb_obs.swapaxes(0,1))


'''reversed返回一个翻转的迭代器'''
# for t in reversed(range(10)):
#     print(t)


'''np.asarray'''
# a = [[1,2],[3,1]]
# print(a)
# a = np.asarray(a)
# print(a)


'''直接赋值，对象的引用'''
# list = [1,2,3,[1,2]]
# b = list
# print(b)
# list.append(1)
# print(b)

'''copy浅拷贝'''
# c = copy.copy(list)
# print(c)
# list.append(5)
# print('list:',list)
# print('c:',c)
#
# print(c[3])
# list[3].append(5)
# print(c[3])

'''copy深拷贝'''
# c = copy.deepcopy(list)
# print(c)
# list.append(5)
# print('list:',list)
# print('c:',c)
#
# print(c[3])
# list[3].append(5)
# print(c[3])
