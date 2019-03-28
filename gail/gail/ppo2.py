import time
# 双向队列
from collections import deque
import random
import numpy as np
import tensorflow as tf
from baselines import logger
from gail import config
from gail.discriminator import Discriminator
from gail.expert import Sampler
from gail.logger import MyLogger

configs = config.configs['gail']
mylogger = MyLogger("./log")

class Model(object):
    def __init__(self, *,sess,policy, ob_space, ac_space, nbatch_act, nbatch_train,
                 nsteps, ent_coef, vf_coef, max_grad_norm):
        # nbatch_act = 8，就等于环境个数nenvs。因为每一次都分别对８个环境执行，得到每个环境中actor的动作。
        # 1为nsteps。其实在CNN中没啥用，在LSTM才会用到（因为LSTM会考虑前nsteps步作为输入）。

        self.global_step_policy = tf.Variable(0, trainable=False)
        mylogger.add_info_txt("Using mlp model")
        act_model = policy(sess, ob_space, ac_space, nbatch_act, 80, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True)
        # act_model = policy(sess, ob_space, ac_space, nbatch=1, nsteps=1, nlstm=256, reuse=False)
        # train_model = policy(sess, ob_space, ac_space, nbatch=4000, nsteps=200, nlstm=256, reuse=True)
        A = train_model.pdtype.sample_placeholder([None])  # action
        ADV = tf.placeholder(tf.float32, [None])  # advantage
        R = tf.placeholder(tf.float32, [None])  # return
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])  # old -logp(action)
        OLDVPRED = tf.placeholder(tf.float32, [None])  # old value prediction
        LR = tf.placeholder(tf.float32, [])  # learning rate
        CLIPRANGE = tf.placeholder(tf.float32, [])
        neglogpac = train_model.pd.neglogp(A)  # -logp(action)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))

        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        # 论文公式(9)，ent_coef, vf_coef分别为PPO论文中的c1, c2，这里分别设为0.01和0.5。entropy为文中的S；pg_loss为文中的L^{CLIP}
        '''This objective can further be augmented by adding an entropy bonus to ensure suﬃcient exploration'''
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        with tf.variable_scope('model'):
            params = tf.trainable_variables()  # 图中需要训练的变量

        grads = tf.gradients(loss, params)  # 计算梯度
        if max_grad_norm is not None:
            # Gradient Clipping的引入是为了处理gradient explosion或者gradients vanishing的问题。当在一次迭代中权重的更新过于迅猛的话，很容易导致loss divergence。
            # Gradient Clipping的直观作用就是让权重的更新限制在一个合适的范围
            # max_grad_norm 是截取的比率
            # 这个函数返回截取过的梯度张量和一个所有张量的全局范数。
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads, global_step=self.global_step_policy)

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):

            advs = returns - values
            '''这里还对adv进行了归一化'''
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X: obs, A: actions, ADV: advs, R: returns, LR: lr,
                      CLIPRANGE: cliprange, OLDNEGLOGPAC: neglogpacs, OLDVPRED: values}

            '''在lstm、rnn时用到'''
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks

            return sess.run([pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],td_map)[:-1]

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
        saver = tf.train.Saver(max_to_keep=10)
        self.save_path = './model/model'

        def save(sess, save_path, global_step):
            saver.save(sess, save_path, global_step)
            # 在模型持久化
            #joblib.dump(ps, save_path)

        def load(sess):
            ckpt = tf.train.get_checkpoint_state('./model/')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                mylogger.add_info_txt("Successfully loaded policy and discriminator model!" +
                                      ckpt.model_checkpoint_path)
            else:
                mylogger.add_info_txt("Could not load any model!")

        '''
        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)
            # If you want to load weights, also save/load observation scaling inside VecNormalize
        '''
        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load

        sess.run(tf.global_variables_initializer())  # pylint: disable=E1101


def rewards_clipping(r):
    return r


class Runner(object):
    def __init__(self, *, sess, env, model, nsteps, gamma, lam):
        self.dloss = 10
        self.gloss = 10
        self.env_global_step = 0  # My note: set by hjf
        self.max_global_steps = 15999  # My note: set by hjf
        self.sess = sess
        self.env = env
        self.model = model
        nenv = env.num_envs
        mylogger.add_info_txt(' env.num_envs: '+str(nenv))
        '''定义obs, We cannot use this obs, c's nums of env is changing'''
        #self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name)  # 16*579
        self.obs = env.obs.copy()  # 注意这里， 不使用copy会使得obs变化时env.obs也变化
        self.agents = env.agents  # List of agents id
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

        '''初始化Discriminator'''
        self.discriminator = Discriminator(arch_params=configs.discriminator_params, stddev=0.02)

        '''定义一些placeholder'''
        self.global_step_dis = tf.Variable(0, trainable=False)
        self.is_training = tf.placeholder(tf.bool)
        self.expert_state = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.env.observation_space.shape))
        # self.expert_action = tf.one_hot(tf.placeholder(dtype=tf.int32, shape=[None]),
        #                                 depth=self.env.action_space.n)  # indices, depth
        self.expert_action = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.env.action_space.shape))

        self.gen_state = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.env.observation_space.shape))
        # self.gen_action = tf.one_hot(tf.placeholder(dtype=tf.int32, shape=[None]),
        #                              depth=self.env.action_space.n)  # indices, depth
        self.gen_action = tf.placeholder(dtype=tf.float32, shape=[None] + list(self.env.action_space.shape))

        '''D网络前向传播的输出值0-1的数值'''
        self.discriminator_expert_output = self.discriminator(self.expert_state, self.expert_action, self.is_training)
        self.discriminator_gen_output = self.discriminator(self.gen_state, self.gen_action, self.is_training,
                                                           reuse=True)

        self.discriminator_loss = - tf.reduce_mean(tf.log(self.discriminator_expert_output + configs.epsilon) + tf.log(
            1 - self.discriminator_gen_output + configs.epsilon))
        self.generator_loss = -tf.reduce_mean(tf.log(self.discriminator_gen_output + configs.epsilon))
        self.discriminator_train_step = tf.train.AdamOptimizer(configs.learning_rate, configs.beta1,
                                                               configs.beta2).minimize(self.discriminator_loss,
                                                                                       var_list=tf.get_collection(
                                                                                           tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                           scope='discriminator'),
                                                                                       global_step=self.global_step_dis)
        tf.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101

    # collect generate experience
    def run(self):
        # mb_agents: my note: a List of agent' ids
        mb_obs, mb_actions, mb_rewards, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_infos = {}  # my note:Dictionary that record  episode of every agent
        epinfos = []
        mb_states = []
        # for _ in range(80):
        #     mb_states.append(self.states.copy())
        if self.states is None:
            mb_states = self.states
        obs_count = 0
        while True:  # My note： collect nsteps*nums data
            temp_count = 0
            while self.obs.shape[0] == 0:
                temp_count += 1
                self.obs, _, self.dones, self.agents, infos = self.env.step(np.zeros([0]))
            if temp_count > 5:
                mylogger.add_warning_txt("no agent in many steps: " + str(temp_count))
            # It's tricky that nums of agents are changing overtime. I need dictionary to
            # record each agent's cell states. But now luckily,  nums of agents is one.
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            logits = self.sess.run(self.discriminator_gen_output,
                                    feed_dict={self.gen_state: np.reshape(self.obs, [-1, 291]),
                                               self.gen_action: np.reshape(actions, [-1, 2]),
                                               self.is_training: False})
            rewards = -np.log(1-logits+1e-8)
            rewards = [rewards_clipping(x) for x in rewards]
            assert len(self.agents) > 0,  "no agent!"
            for i in range(len(self.agents)):
                aid = self.agents[i]
                if aid not in mb_infos.keys():
                    mb_infos[aid] = []
                mb_infos[aid].append([self.obs[i], actions[i], rewards[i], values[i], self.dones[i],
                                      neglogpacs[i]])

            obs_count += len(self.dones)
            if obs_count >= self.nsteps * 80:
                break
            temp_count = 0
            while self.obs.shape[0] == 0:
                temp_count += 1
                self.obs, _, self.dones, self.agents, infos = self.env.step(np.zeros([0]))
            if temp_count > 5:
                mylogger.add_warning_txt("no agent in many steps: "+str(temp_count))
            self.obs, _, self.dones, self.agents, infos = self.env.step(actions)
            self.env_global_step = (self.env_global_step+1) % self.max_global_steps
        # 将上面的数据压成nsteps*4长的列表
        all_eps = []
        for key in mb_infos.keys():
            all_eps += mb_infos[key]
        mylogger.add_info_txt("平均轨迹长度为： "+str(16000/len(mb_infos)))
        assert len(all_eps) >= self.nsteps*80, "no enough states"
        for _ in range(len(all_eps)-self.nsteps*80):
            all_eps.pop()

        for e in all_eps:
            mb_obs.append(e[0])  # e[0].shape: (579,)
            mb_actions.append(e[1])
            mb_rewards.append(e[2])
            mb_values.append(e[3])
            mb_dones.append(e[4])
            mb_neglogpacs.append(e[5])

        mb_obs = np.asarray(mb_obs, dtype=np.float32).reshape([self.nsteps, 80, 291])
        # mb_rewards = (mb_rewards-np.mean(mb_rewards))/(np.std(mb_rewards)+1e-8)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).reshape([self.nsteps, 80])
        mb_actions = np.asarray(mb_actions, np.float32).reshape([self.nsteps, 80, 2])
        mb_values = np.asarray(mb_values, dtype=np.float32).reshape([self.nsteps, 80])
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32).reshape([self.nsteps, 80])
        mb_dones = np.asarray(mb_dones, dtype=np.bool).reshape([self.nsteps, 80])
        last_values = self.model.value(self.obs, self.states, self.dones)
        # mb_states = np.asarray(mb_states, dtype=np.float32).reshape([80, 512])
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                '''为什么？'''
                nextnonterminal = 1.0 - self.dones[0]  # false:0.0, true:1.0
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1][0]
                nextvalues = mb_values[t + 1]


            # print(mb_rewards.shape)
            # print(mb_values.shape)
            # print(mb_rewards[t])
            # print(mb_values[t])
            # My note: if tras done
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]

            # print(delta)
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

        mb_returns = mb_advs + mb_values

        '''第一个参数接受一个函数名，后面的参数接受一个或多个可迭代的序列，返回的是一个集合'''
        '''将所有的调用的结果作为一个list返回。如果func为None，作用同zip()。'''
        '''返回的是一个tuple'''
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
                mb_states, epinfos)

    def train_discriminator(self, expert_state, expert_action, gen_state, gen_action, is_training=True):
        _, dloss_curr, gloss_cur = self.sess.run([self.discriminator_train_step, self.discriminator_loss, self.generator_loss],
                                      feed_dict={self.expert_state: expert_state,
                                                 self.expert_action: expert_action,
                                                 self.gen_state: gen_state,
                                                 self.gen_action: gen_action,
                                                 self.is_training: is_training})
        return dloss_curr, gloss_cur

    def eval_gloss(self, gen_state, gen_action, is_training=True):
        gloss_cur = self.sess.run(self.generator_loss,
                                  feed_dict={self.gen_state: gen_state,
                                             self.gen_action: gen_action,
                                             self.is_training: is_training})
        return gloss_cur


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    # 将指定的轴对换了 swap and then flatten axes 0 and 1
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def constfn(val):
    def f(_):
        return val

    return f


def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
          vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
          save_interval=0):
    '''
        Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

        Parameters:
        ----------

        network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                          specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                          tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                          neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                          See common/models.py/lstm for more details on using recurrent nets in policies

        env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                          The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


        nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                          nenv is number of environment copies simulated in parallel)

        total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

        ent_coef: float                   policy entropy coefficient in the optimization objective

        lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                          training and 0 is the end of the training.

        vf_coef: float                    value function loss coefficient in the optimization objective

        max_grad_norm: float or None      gradient norm clipping coefficient

        gamma: float                      discounting factor

        lam: float                        advantage estimation discounting factor (lambda in the paper)

        log_interval: int                 number of timesteps between logging events

        nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                          should be smaller or equal than number of environments run in parallel.

        noptepochs: int                   number of training epochs per update

        cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                          and 0 is the end of the training

        save_interval: int                number of timesteps between saving events

        load_path: str                    path to load the model from

        **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                          For instance, 'mlp' network architecture has arguments num_hidden and num_layers.

        '''
    if isinstance(lr, float):
        lr = constfn(lr)
    else:
        assert callable(lr)  # 方法用来检测对象是否可被调用
    if isinstance(cliprange, float):
        cliprange = constfn(cliprange)
    else:
        assert callable(cliprange)

    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    nenvs = 80
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches  # 整除
    mylogger.add_info_txt('nbatch, nbatch_train, nminibatches: '+str(nbatch)+','+str(nbatch_train)+','+str(nminibatches))
    '''定义了策略模型和值函数模型'''
    sess = tf.Session()
    make_model = lambda: Model(sess=sess,policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
                               nbatch_train=nbatch_train,
                               nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                               max_grad_norm=max_grad_norm)

    # save_interval 保存的时间间隔
    '''
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    '''
    model = make_model()  # make two model. act_model and train_model
    runner = Runner(sess=sess, env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    model.load(sess=sess)
    mylogger.add_sess_graph(sess.graph)
    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()
    # model_path = '/home/zhangkaifeng/Breakout_gail_2/'
    sampler = Sampler()
    # 模型更新次数
    nupdates = total_timesteps // nbatch
    mylogger.add_info_txt('nupdates: '+str(nupdates))
    old_gloss = 1.5
    accumulate_improve = 0
    totalsNotUpdateG = 0
    obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()
    states_expert, actions_expert = sampler.next_batch_samples_v1(configs.batch_size, runner.env_global_step)
    policy_step = sess.run(model.global_step_policy)
    for update in range(1, nupdates + 1):
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)

        mylogger.add_info_txt('=========================================')
        mylogger.add_info_txt('lr of policy model now: '+str(lrnow))
        mylogger.write_summary_scalar(policy_step//noptepochs//nminibatches, 'lrG', lrnow)
        assert nbatch % nminibatches == 0
        if totalsNotUpdateG > 50 and accumulate_improve <= 0:
            accumulate_improve = 1
            totalsNotUpdateG = 0
            np.savetxt('obs.txt', obs, fmt='%10.6f')

        # print('obs', obs.shape, obs[0][:10])
        #         # print('return.shape', returns.shape, returns[0])
        #         # print('masks_action',masks.shape, masks[0])
        #         # print('actions', actions.shape, actions[0])
        #         # print('values', values.shape[0], values[0])
        #         # print('neglogpacs', neglogpacs.shape, neglogpacs[0])
        # sampler.next_buffers(env_global_step=runner.env_global_step)
        inds = np.arange(nbatch)
        np.random.shuffle(inds)
        # for iter in range(configs.discriminator_update):
        for iter in range(1):  # update discriminator
            # indxs = random.sample(range(runner.nsteps * 16), configs.batch_size)
            if accumulate_improve < 0.05 and update > 20:  # policy have little improve
                break
            mylogger.add_info_txt('***********update discriminator and reset accumulate_improve=0***************')
            accumulate_improve = 0
            start = iter*configs.batch_size
            end = start+configs.batch_size
            indxs = inds[start:end]
            # states_expert, actions_expert = sampler.next_batch_samples(configs.batch_size, indxs)
            obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()
            # obs = obs + (np.random.normal(0, 0.2, 16000*291) * (np.exp(-policy_step / 100))).reshape(obs.shape)
            mean_ret = np.mean(returns)
            mean_values = np.mean(values)
            mean_adv = np.mean(returns - values)
            dis_step = sess.run(runner.global_step_dis)
            mylogger.write_summary_scalar(dis_step, "mean_values", mean_values)
            mylogger.write_summary_scalar(dis_step, "mean_adv", mean_adv)
            mylogger.write_summary_scalar(dis_step, "mean_ret", mean_ret)
            mylogger.add_info_txt('第' + str(update) + '大更新， 估计的回报为：' + str(mean_ret))
            mylogger.add_info_txt('第' + str(update) + '大更新， 估计的值函数为：' + str(mean_values))
            mylogger.add_info_txt('第' + str(update) + '大更新， 估计的优势函数为：' + str(mean_adv))
            states_expert, actions_expert = sampler.next_batch_samples_v1(configs.batch_size, runner.env_global_step)
            # states_expert += (np.random.normal(0, 0.2, 16000*291) * (np.exp(-policy_step / 100))).reshape(obs.shape)
            # print('states_expert.shape, actions_expert.shape', states_expert.shape, actions_expert.shape)
            states_gen_tmp, actions_gen_tmp = obs.reshape(runner.nsteps*80, 291)[indxs, :],\
                                              actions.reshape(runner.nsteps*80, 2)[indxs, :]

            runner.dloss, runner.gloss = runner.train_discriminator(states_expert,
                                               actions_expert,
                                               states_gen_tmp,
                                               actions_gen_tmp)
            # I have try to place this code outside of 'for' block, also get a good result,
            # but discriminator update rarely
            old_gloss = runner.eval_gloss(obs.reshape([runner.nsteps*80, 291]),
                                          actions.reshape([runner.nsteps*80, 2]))

        # my note: 'mb_returns = mb_advs + mb_values', return的作用是什么？
        # obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()

        epinfobuf.extend(epinfos)
        mblossvals = []
        if states is None:  # nonrecurrent version
            for i in range(2):  # critic part of policy is 2
                inds = np.arange(nbatch)
                obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()
                # obs = obs + (np.random.normal(0, 0.2, 16000*291) * (np.exp(-policy_step/100))).reshape(obs.shape)
                for _ in range(noptepochs):  # noptepochs = 4
                    '''why shuffle?'''
                    np.random.shuffle(inds)
                    for start in range(0, nbatch, nbatch_train):  # my note:  iteration,nbatch,nbatch_trai=4,1024*16,4096
                        end = start + nbatch_train
                        mbinds = inds[start:end]
                        slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        mblossvals.append(model.train(lrnow, cliprangenow, *slices))
                new_gloss = runner.eval_gloss(obs.reshape([runner.nsteps * 80, 291]),
                                              actions.reshape([runner.nsteps * 80, 2]))

        else:  # recurrent version
            for i in range(2):  # critic part of policy if 2
                obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()
                # obs = obs + (np.random.normal(0, 0.2, 16000 * 291) * (np.exp(-policy_step / 100))).reshape(obs.shape)
                # print('states.shape', states.shape)
                assert nenvs % nminibatches == 0
                envsperbatch = nenvs // nminibatches
                envinds = np.arange(nenvs)
                flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
                envsperbatch = nbatch_train // nsteps
                for _ in range(noptepochs):
                    np.random.shuffle(envinds)
                    for start in range(0, nenvs, envsperbatch):
                        end = start + envsperbatch
                        mbenvinds = envinds[start:end]
                        mbflatinds = flatinds[mbenvinds].ravel()  # ravel: Flatten. Get some env's step.

                        slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        mbstates = states[mbenvinds]
                        mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))
                new_gloss = runner.eval_gloss(obs.reshape([runner.nsteps * 80, 291]),
                                              actions.reshape([runner.nsteps * 80, 2]))
                accumulate_improve += old_gloss - new_gloss
        totalsNotUpdateG += 1
        mylogger.add_info_txt('old gloss：'+str(old_gloss)+'; new gloss: '+str(new_gloss)+
                              '; accumulate_improve: '+str(accumulate_improve))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))

        policy_step = sess.run(model.global_step_policy)
        mean_monte_reward = np.mean(-np.log(1 - np.exp(-runner.gloss)))
        mylogger.add_info_txt('第' + str(update) + '次读取专家数据成功; '+'第' + str(update) + '次生成数据成功; '+
                              "monte carlo return: " + str(mean_monte_reward))
        mylogger.add_info_txt('第' + str(update) + '次读取专家数据成功; '+'第' + str(update) + '次生成数据成功; '+
                              "discriminator loss: " + str(runner.dloss))
        mylogger.add_info_txt('第' + str(update) + '次读取专家数据成功; '+'第' + str(update) + '次生成数据成功; '+
                              "generator loss: "+ str(runner.gloss))

        mylogger.write_summary_scalar(policy_step//(noptepochs*nminibatches),
                                      "generator_loss", runner.gloss)  # update == policy_step//noptepochs
        mylogger.write_summary_scalar(policy_step//(noptepochs*nminibatches),
                                      "mean_monte_reward", mean_monte_reward)
        mylogger.write_summary_scalar(policy_step//(noptepochs*nminibatches), "dloss", runner.dloss)
        '''pg_loss, vf_loss, entropy'''

        mylogger.write_summary_scalar(policy_step//(noptepochs*nminibatches), "pg_loss", lossvals[0])
        mylogger.write_summary_scalar(policy_step//(noptepochs*nminibatches), "vf_loss", lossvals[1])
        mylogger.write_summary_scalar(policy_step//(noptepochs*nminibatches), "entropy", lossvals[2])

        # logger.logkv("dloss", runner.dloss)
        # logger.logkv("lossvals", lossvals)
        # if update % log_interval == 0 or update == 1:
        #     ev = explained_variance(values, returns)  # Returns 1 - Var[y-ypred] / Var[y]
        #     mylogger.write_summary_scalar(update, "serial_timesteps", update * nsteps)
        #     mylogger.write_summary_scalar(update, "total_timesteps", update * nbatch)
        #     mylogger.write_summary_scalar(update, "fps", fps)
        #     mylogger.write_summary_scalar(update, "explained_variance", float(ev))
        #     mylogger.write_summary_scalar(update, "eprewmean", safemean([epinfo['r'] for epinfo in epinfobuf]))
        #     mylogger.write_summary_scalar(update, "eplenmean", safemean([epinfo['l'] for epinfo in epinfobuf]))
        #     mylogger.write_summary_scalar(update, "time_elapsed", tnow - tfirststart)
        #     logger.logkv("serial_timesteps", update * nsteps)
        #     logger.logkv("nupdates", update)
        #     logger.logkv("total_timesteps", update * nbatch)
        #     logger.logkv("fps", fps)
        #     logger.logkv("explained_variance", float(ev))
        #     logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
        #     logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
        #     logger.logkv('time_elapsed', tnow - tfirststart)
        #
        #     for (lossval, lossname) in zip(lossvals, model.loss_names):
        #         logger.logkv(lossname, lossval)
        #     logger.dumpkvs()

    #    if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
    #        checkdir = osp.join(logger.get_dir(), 'checkpoints')
    #        os.makedirs(checkdir, exist_ok=True)
    #        savepath = osp.join(checkdir, '%.5i' % update)
    #        print('Saving to', savepath)
    #        model.save(savepath)
        mylogger.add_info_txt('save_interval'+str(save_interval)+'update'+str(update))
        if save_interval and (policy_step//(noptepochs*nminibatches) % save_interval == 0 and
                              policy_step % noptepochs == 0 or update == 1):
            mylogger.add_info_txt("saved ckpt model!")
            model.save(sess=sess, save_path=model.save_path, global_step=policy_step//(noptepochs*nminibatches))
    np.savetxt('obs'+str(update)+'.txt', obs, fmt='%10.6f')
    env.close()


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


