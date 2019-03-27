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
        self.states = np.loadtxt('C:/Users/eatAlot/Desktop/第二学期/工作/gail训练交接/gail/gail/data/replay_data/new_state72.txt')
        self.actions = np.loadtxt('C:/Users/eatAlot/Desktop/第二学期/工作/gail训练交接/gail/gail/data/replay_data/actions.txt')
        self.frames = np.loadtxt('C:/Users/eatAlot/Desktop/第二学期/工作/gail训练交接/gail/gail/data/replay_data/frames.txt')[:, 0]

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
        self.positions = np.loadtxt('C:/Users/eatAlot/Desktop/第二学期/工作/gail训练交接/gail/gail/data/replay_data/frames.txt')
        assert self.positions.shape[0] == self.states.shape[0]
        self.cycle_len = self.positions.shape[0]
        self.cycle = 0
        self.idx_list = []
        self.idx = 0
        self.chosen_id = 0
        self.end_frame = 0
        self.start = True
        self.count = 0

    # 这里其实还可以继续优化
    def next_buffers(self, env_global_step):
        # 一直找到一个刚好等于env_global_step 的frame， 如果不存在， 则转向0
        frame_begin = 0
        idxs = []
        for i in range(self.frames.shape[0]):
            if self.frames[i] < env_global_step+2:
                continue
            else:
                frame_begin = i
                break
        for i in range(1024*16):
            idxs.append((i+frame_begin) % self.frames.shape[0])
        self.states_buffer[:] = self.states[idxs, :]
        self.actions_buffer[:] = self.actions[idxs, :]

    def next_batch_samples(self, batch_size, idxs):
        # idxs = random.sample(range(1024*16), batch_size)
        return (self.states_buffer[idxs, :], self.actions_buffer[idxs, :])

    def choose_rule(self, cycle, size):
        if size == 0:
            return 0
        else:
            return cycle % size

    # need to consider special cases: length of epi = 1
    def choose_person(self):
        self.count += 1
        size = 0
        frame = self.positions[self.idx][0]
        while self.positions[self.idx][0] == frame:
            self.idx += 1
            size += 1
        chosen_id = self.choose_rule(self.cycle, size)
        self.chosen_id = self.positions[self.idx + chosen_id - size][2]  # chose a people
        self.end_frame = self.positions[self.idx + chosen_id - size][1]
        # print(self.chosen_id, self.end_frame, self.idx)
        # add experience index to idx_list
        self.idx_list.append(self.idx + chosen_id - size)
        if self.end_frame == self.positions[self.idx + chosen_id - size][0]:
            # print('abn', self.chosen_id, self.end_frame, self.idx)
            self.choose_person()

    def next_batch_samples_v1(self, bath_size, env_global_step):
        # if current chosen episode ending, etc. current frame > end_frame, chose another id
        self.idx_list = []
        # find util the chosen people disppear
        if self.start:
            self.start = False
            self.choose_person()
        while True:
            if self.idx == 0:  # this cycle ended
                print('\nself.cycle', self.cycle)
                # self.cycle += 1 # if we want have more diverse trace, we can set different cycle
            # collect enough data, so redirect self.idx to beginning of next
            # frame and return the buffer and reset self.idx_list
            if len(self.idx_list) == 80*200:
                while self.positions[self.idx][0] == self.end_frame:  # redirect self.idx to next frame
                    self.idx = (self.idx + 1) % self.cycle_len
                return self.states[self.idx_list, :], self.actions[self.idx_list, :]

            # belong to current people's episode
            if (self.positions[self.idx][2] == self.chosen_id) and \
                    (self.positions[self.idx][1] > self.positions[self.idx][0]):
                self.idx_list.append(self.idx)
            self.idx = (self.idx+1) % self.cycle_len

            # new person episode, if len(episode)=1, will be wrong
            if (self.positions[self.idx][2] == self.chosen_id) and \
                    (self.positions[self.idx][1] == self.positions[self.idx][0]):
                while self.positions[self.idx][0] == self.end_frame:  # redirect self.idx to next frame
                    self.idx = (self.idx + 1) % self.cycle_len
                if self.idx == 0:  # this cycle ended
                    # self.cycle += 1
                    pass
                self.choose_person()


def test_v1():
    sampler = Sampler()
    for i in range(100):
        state_buffer, action_buffer = sampler.next_batch_samples_v1(1024, 1)
        print('tate_buffer.shape, action_buffer.shape', state_buffer.shape, action_buffer.shape)
        print('sampler.cycle, sampler.end_frame', sampler.cycle, sampler.end_frame)
        print('sampler.count', sampler.count)
        np.savetxt('test_state.txt', state_buffer, fmt='%10.6f')
        np.savetxt('test_acs.txt', action_buffer, fmt='%10.6f')
