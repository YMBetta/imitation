
import sys
from baselines import logger
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
import multiprocessing
import tensorflow as tf
import os
import unity_env
from policies import CnnPolicy, LstmPolicy, LnLstmPolicy,MlpPolicy
import ppo2


#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train(env, num_timesteps, seed, policy):

    ncpu = multiprocessing.cpu_count()
    #sys.platform 该变量返回当前系统的平台标识。
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True, #自动选择运行设备
                            intra_op_parallelism_threads=ncpu, #控制运算符op内部的并行
                            inter_op_parallelism_threads=ncpu) #控制多个运算符op之间的并行计算
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    tf.Session(config=config).__enter__()

    policy = {'mlp':MlpPolicy, 'cnn' : CnnPolicy, 'lstm' : LstmPolicy, 'lnlstm' : LnLstmPolicy}[policy]

    ppo2.learn(policy=policy, env=env, nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=lambda f : f * 0.1,
        total_timesteps=int(num_timesteps * 1.1))

def main():
    env = unity_env.UnityEnv()
    num_timesteps = 1e8
    policy = 'mlp'
    print(num_timesteps)
    train(env, num_timesteps=num_timesteps, seed=None, policy=policy)

if __name__ == '__main__':
    main()
    logger.configure()