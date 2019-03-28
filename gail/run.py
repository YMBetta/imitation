
import numpy as np
import gym
import os
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from mlagents.envs import UnityEnvironment
from gail import policies
from baselines.ddpg import ddpg
from baselines.common import models
# from
from gail import ppo2
import tensorflow as tf

class UnityEnv():
    def __init__(self,episode_len=1000000):

        # work_id 即端口
        self.env = UnityEnvironment(file_name='D:/Unity/unity_workspace/train_Gail_RrainReplay'
                                              '/train_gail_replay/replayDull/train_Gail', worker_id=9000, seed=1)
        # self.env = UnityEnvironment(file_name=None, worker_id=0, seed=1)
        '''获取信息'''
        self.brain_name = self.env.brain_names[0]
        print('brain_name:', self.brain_name)
        self.env.reset()
        info = self.env.step()
        brainInfo = info[self.brain_name]

        '''设置动作、观测空间'''
        #self.action_space = spaces.Discrete(1)
        #self.action_space = spaces.Tuple([spaces.Discrete(2),spaces.Discrete(2)])
        #self.action_space  = spaces.MultiDiscrete([2,2])
        # self.action_space      = spaces.Box(low=np.array([-2, -2]),    high=np.array([+1,+1]),  dtype=int)
        # self.observation_space = spaces.Box(low=np.array([-1,-1,-1,-1,-1,-1,-1,-1]), high=np.array([1,1,1,1,1,1,1,1]),dtype=np.float32)

        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([+1, +20]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.zeros([291]) - 10, high=np.zeros([291]) + 10, dtype=np.float32)
        self.obs = brainInfo.vector_observations.copy()  # two dimensional numpy array
        self.agents = brainInfo.agents
        self.num_envs = len(self.agents)  # num of agents
        count = 0
        while not self.obs.shape[0]:
            count += 1
            print('init steps:', count)
            info = self.env.step()
            brainInfo = info[self.brain_name]
            self.obs = brainInfo.vector_observations
            self.agents = brainInfo.agents
            self.num_envs = len(self.agents)  # num of agents
        self.num_steps = 0
        self.seed()
        self.episode_len = episode_len

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    def step(self, a):  # step in environment
        action = {}
        '''a 的个数为nums_envs*dim（action）'''
        action[self.brain_name] = a
        info = self.env.step(vector_action=action)
        brainInfo = info[self.brain_name]

        reward = np.array(brainInfo.rewards)
        done = np.array(brainInfo.local_done)  # local_done: type:list, length:num of agents
        ob = np.array(brainInfo.vector_observations)
        agents = brainInfo.agents
        self.num_steps += 1
        # print(' action：',a[0], ' reward：',reward,' num_steps：',self.num_steps)
        return ob, reward, done, agents, {}

    def reset(self):  # reset environment
        self.env.reset()
        info = self.env.step()
        brainInfo = info[self.brain_name]
        return np.array(brainInfo.vector_observations)  # return 2D numpy array

    def render(self):
        pass

    def close(self):
        self.env.close()

def gail():
    num_timesteps = 10000000  # 1*1e8
    env = UnityEnv()
    ppo2.learn(policy=policies.MlpPolicy,
               env=env,
               nsteps=200,
               total_timesteps=num_timesteps,
               ent_coef=1e-3,
               lr=3e-4,
               vf_coef=0.5,
               max_grad_norm=0.5,
               gamma=0.99,
               lam=0.95,
               log_interval=10,
               nminibatches=4,
               noptepochs=4,
               cliprange=0.2,
               save_interval=50)

def DDPG():
    num_timesteps = 1000000  # 1e6
    env = UnityEnv()
    ddpg.learn(network ='mlp',
               env=env,
               total_timesteps=None,
               nb_epochs=None,  # with default settings, perform 1M steps total
               nb_epoch_cycles=20,
               nb_rollout_steps=100,
               reward_scale=1.0,
               render=False,
               render_eval=False,
               noise_type='adaptive-param_0.2',
               normalize_returns=False,
               normalize_observations=True,
               critic_l2_reg=1e-2,
               actor_lr=1e-4,
               critic_lr=1e-3,
               popart=False,
               gamma=0.99,
               clip_norm=None,
               nb_train_steps=50,  # per epoch cycle and MPI worker,
               nb_eval_steps=100,
               batch_size=64,  # per MPI worker
               tau=0.01,
               eval_env=None,
               param_noise_adaption_interval=50,
               )

def main():
    #logger.configure(dir='/Users/liuyawen/Desktop/项目/bullet3')
    gail()
    # for i in range(100):
    #     print(i)

    #DDPG()
    #results_plotter.main()


if __name__ == '__main__':
    main()