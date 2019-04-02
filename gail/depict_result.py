#!/usr/bin/env python
# encoding: utf-8
'''
@author: Huang Junfu
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 2504598262@qq.com
@software: 
@file: depict_result.py
@time: 2019/4/1 7:37
@desc:
'''

from mpl_toolkits.mplot3d import Axes3D
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
PI = np.pi


def depict_action(action, color):
    plt.scatter(action[:, 0], action[:, 1], c=color)


def depict_obs(obs, color):
    plt.scatter(obs[:, 2], obs[:, 1], c=color)
    plt.xlabel('angle')
    plt.ylabel('logdist')


def main():
    # gen_obs = np.loadtxt('obs.txt')[:2000]
    # depict_obs(gen_obs, color='r')
    real_obs = np.loadtxt('sortedData_v4/Xtrain.txt')[:2000, 5:8]
    depict_obs(real_obs, color='r')
    plt.show()


if __name__ == '__main__':
    main()
