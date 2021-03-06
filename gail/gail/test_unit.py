#!/usr/bin/env python
# encoding: utf-8
'''
@author: Huang Junfu
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 2504598262@qq.com
@software: 
@file: tbTest.py
@time: 2019/2/20 12:25
@desc:
'''
import pytest
import gym

common_kwargs = dict(
    total_timesteps=3000,
    network='mlp',
    gamma=1.0,
    seed=0,
)

learn_kwargs = {
    'a2c' : dict(nsteps=32, value_network='copy', lr=0.05),
    'acer': dict(value_network='copy'),
    'acktr': dict(nsteps=32, value_network='copy', is_async=False),
    'deepq': dict(total_timesteps=20000),
    'ppo2': dict(value_network='copy'),
    'trpo_mpi': {}
}

@pytest.mark.slow
@pytest.mark.parametrize("alg", learn_kwargs.keys())
def test_cartpole(alg):
    '''
    Test if the algorithm (with an mlp policy)
    can learn to balance the cartpole
    '''

    kwargs = common_kwargs.copy()
    kwargs.update(learn_kwargs[alg])

    learn_fn = lambda e: get_learn_function(alg)(env=e, **kwargs)
    def env_fn():

        env = gym.make('CartPole-v0')
        env.seed(0)
        return env

    reward_per_episode_test(env_fn, learn_fn, 100)

if __name__ == '__main__':
    test_cartpole('acer')

