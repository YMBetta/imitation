import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from gym.utils import seeding
from gym import error, spaces


def compute_distance(current, target):
    return (np.sum((current-target)**2))**.5


def target_angle(pre, current, target):
    """compute the angle of target"""
    pre_curr_dis = current - pre
    curr_tar_dis = target - current
    rtn = compute_angle(curr_tar_dis[0], curr_tar_dis[1]) - \
          compute_angle(pre_curr_dis[0], pre_curr_dis[1])
    if rtn > np.pi:
        rtn -= 2*np.pi
    if rtn < -np.pi:
        rtn += 2*np.pi
    return rtn


def compute_angle(x, y):  # dis=(x,y)
    if x == 0.:
        if y > 0.:
            return np.pi/2
        elif y == 0.:
            return 0.
        else:
            return -np.pi/2
    else:
        r = y/x
        arctan = np.arctan(r)
        if arctan > 0.:
            if x > 0.:
                return arctan
            else:
                return arctan - np.pi
        elif arctan == 0.:
            if x > 0.:
                return 0
            else:
                return np.pi
        else:
            if x > 0.:
                return arctan
            else:
                return np.pi + arctan


def Qualine(t, control_point):  # control point: R^2,  >0, <1; t: >0, <1
    return t*np.array([1, 0])+(0.1*t - 0.1*t**2)*control_point


def generate(nums_tras, nsteps):
    y = np.ones([nsteps, 2])
    x = np.arange(0, 1, 1/nsteps)
    all_points = []
    tras = []

    for _ in range(nums_tras):
        control_point = (np.random.random(2)-0.5)*20
        all_points.append(control_point)
        tra = []
        for i in range(nsteps):
            t = x[i]
            y[i, :] = Qualine(t, control_point)
            tra.append(y[i].copy())
        tra.append([1, 0])
        tras.append(tra)
        plt.plot(y[:, 0], y[:, 1])
    # obs: (tar_angle, tar_dis)
    tras = np.array(tras)
    np.savetxt('tras_pos.txt', tras.reshape([-1, 2]))
    print('tras.shape', tras.shape)
    assert tras.shape.__len__() == 3
    all_acs = []
    all_obs = []
    tar = np.array([1, 0])
    for n in range(nums_tras):  # nth tras
        pre = curr = tras[n, 0, :]
        acs = []
        obs = []
        for i in range(nsteps):
            curr = tras[n, i, :]  # ith step
            next_pos = tras[n, i+1, :]
            tar_dis = compute_distance(curr, tar)
            tar_angle = target_angle(pre, curr, tar)
            obs.append([tar_angle, tar_dis])
            movement = compute_distance(curr, next_pos)
            move_angle = target_angle(pre, curr, next_pos)
            acs.append([move_angle, movement])
            pre = curr
        obs.append([0, 0])
        acs.append([0, 0])
        all_acs.append(acs)
        all_obs.append(obs)
    all_obs = np.array(all_obs)
    all_acs = np.array(all_acs)
    print(all_obs.shape, all_acs.shape)
    np.savetxt('obs.txt', all_obs.reshape([-1, 2]))
    np.savetxt('acs.txt', all_acs.reshape([-1, 2]))
    np.savetxt('control_points.txt', X=np.array(all_points))
    plt.show()


# generate(nums_tras=10, nsteps=100)


def test_generate():
    acs = np.loadtxt('acs.txt').reshape(10, 101, 2)[8, :, :]

    angle = 0
    pre = curr = np.zeros(2)
    tras = []
    for i in range(100):
        angle += acs[i, 0]
        x = acs[i, 1]*np.cos(angle)
        y = acs[i, 1]*np.sin(angle)
        curr[0] += x
        curr[1] += y
        tras.append(curr.copy())
    tras = np.array(tras)
    print('tras.shape', tras.shape)
    plt.plot(tras[:, 0], tras[:, 1])
    plt.show()


class Env2d:
    def __init__(self):
        self.pre_pos = np.zeros(2)
        self.cur_pos = np.zeros(2)
        self.tar_pos = np.array([1, 0])
        self.world_angle = 0.
        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([+1, +1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.zeros([2]) - 1, high=np.zeros([2]) + 1, dtype=np.float32)
        self.obs = np.array([0, 1])

    def reset(self):
        self.world_angle = 0.
        self.cur_pos = np.zeros(2)

    def reward(self):
        pass

    def step(self, action):
        """return obs and done"""
        self.world_angle += action[0]
        self.pre_pos = self.cur_pos.copy()
        self.cur_pos[0] += action[1]*np.cos(action[0])
        self.cur_pos[1] += action[1]*np.sin(action[0])
        tar_angle = target_angle(self.pre_pos, self.cur_pos, self.tar_pos)
        tar_dist = compute_distance(self.cur_pos, self.tar_pos)
        return np.array([tar_angle, tar_dist]), False


def test_env():
    data = []
    dones = []
    ac = np.array([0, 0.01])
    env = Env2d()
    env.reset()
    for i in range(100):
        ac = np.random.random(2)*2
        ac[1] = 0.01
        if i == 1:
            print(ac)
        obs, done = env.step(action=ac)
        data.append(env.cur_pos.copy())
        dones.append(done)
    print(dones)
    data = np.array(data)
    plt.plot(data[:, 0], data[:, 1])
    plt.show()

