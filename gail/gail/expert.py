import numpy as np
#argparse是python标准库里面用来处理命令行参数的库
import argparse
import random
import os
import time
import numpy as np
#pickle: 用于python特有的类型和python的数据类型间进行转换
import pickle
#处理csv文件
import csv


class Sampler(object):
    def __init__(self):
        self.data_size = 0
        self.count = 0
        self.states = np.loadtxt('C:/Users/eatAlot/Desktop/第二学期/工作/'
                                 'gail训练交接/gail/gail/data/replay_data/seq_state72.txt')
        self.actions = np.loadtxt('C:/Users/eatAlot/Desktop/第二学期/工作/'
                                  'gail训练交接/gail/gail/data/replay_data/seq_actions.txt')
        self.norm_max = np.ones(291)  # which dimesion in obs should be normalize: obs/norm_max.
        self.actions_max = np.ones(2)
        # self.states = np.zeros([self.frames.shape[0], 291])
        # self.actions = np.zeros([self.frames.shape[0], 2])
        # self.mean = np.loadtxt('gail/data/mean.txt')
        # self.std = np.loadtxt('gail/data/std.txt')
        # self.states_buffer = np.zeros([1024*16, 579])
        # self.actions_buffer = np.zeros([1024*16, 2])
        self.mean = 0.
        self.std = 1.
        self.data_size = len(self.actions)

        # testing variables
        self.total_steps = (self.actions.shape[0] // 16000)*16000
        self.cycle = 0
        self.idx_list = []
        self.idx = 0
        self.chosen_id = 0
        self.end_frame = 0
        self.start = True
        self.count = 0

    def do_norm_max(self):
        norm_dims = []
        for i in range(30):
            norm_dims.append(i)
        for i in range(33, 318, 4):
            norm_dims.append(i)
        self.norm_max[:] = np.max(self.states, axis=0)
        for i in range(318):
            if i not in norm_dims:
                self.norm_max[i] = 1
        # self.norm_max = np.ones(318)
        # self.actions_max = np.ones(2)
        self.states = self.states/self.norm_max
        self.actions_max[:] = np.max(self.actions, axis=0)[:2]
        self.actions = self.actions/np.max(self.actions, axis=0)
        print(self.actions_max)
        print(self.norm_max)
        
    def next_batch(self):

        batch_states = self.states[self.idx:self.idx+16000, :]
        batch_actions = self.actions[self.idx:self.idx+16000, :]
        assert batch_actions.shape[0] == batch_states.shape[0]
        self.idx += 16000
        if self.idx >= self.total_steps:
            self.idx = 0

        return batch_states, batch_actions
