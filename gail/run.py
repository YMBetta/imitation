
import numpy as np
from gym import error, spaces
from env.env import Env2d
from gym import utils
from gym.utils import seeding
from gail import policies
from baselines.ddpg import ddpg
from baselines.common import models
# from
from gail import ppo2
import tensorflow as tf

def gail():
    num_timesteps = 100000  # 1*1e7
    env = Env2d()
    ppo2.learn(policy=policies.MlpPolicy,
               env=env,
               nsteps=100,
               total_timesteps=num_timesteps,
               ent_coef=1e-3,
               lr=1e-5,
               vf_coef=0.5,
               max_grad_norm=1,
               gamma=0.99,
               lam=0.95,
               log_interval=10,
               nminibatches=1,
               noptepochs=1,
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