#!/usr/bin/env python
# encoding: utf-8
'''
@author: Huang Junfu
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 2504598262@qq.com
@software: 
@file: temp.py
@time: 2019/2/17 19:44
@desc:
'''
import time
import os
import datetime
import numpy as np
from mlagents.envs import UnityEnvironment
import os
starttime = datetime.datetime.now()
# env = UnityEnvironment(file_name='D:/Unity/unity_workspace/train_Gail_RrainReplay'
#                                               '/train_gail_replay/replayDull/train_Gail', worker_id=9000, seed=1)
env = UnityEnvironment(file_name=None, worker_id=0, seed=1)
env.reset()
# os.system("pause")
obs = []
dones = []
dones_Train_gail = []
brain_name = env.brain_names[0]
info = env.step()
info = env.step()
info = env.step()
brain_info = info[brain_name]
ob = brain_info.vector_observations
print(ob.shape)
for i in range(16000):
    action = {brain_name: np.zeros([1, 2])}
    empty_count = 0
    while ob.shape[0] == 0:
        info = env.step(vector_action={brain_name: np.zeros([0])})
        brain_info = info[brain_name]
        ob = brain_info.vector_observations
        empty_count += 1
    if empty_count > 1:
        print('empty_count: ', empty_count)
    info = env.step(action)
    brain_info = info[brain_name]
    ob = brain_info.vector_observations
    done = brain_info.local_done
    dones_Train_gail.append(done)
# info = env.step()
# braininfo = info['Train_gail']
# obs.append(braininfo.vector_observations)
# dones.append(braininfo.local_done)
# for i in range(16000):
#     time.sleep()
#     env.step(np.ones([1,2])/10)
    # if braininfo.vector_observations.shape[0] == 0:
    #     action = {'Train_gail': np.zeros([0])}
    #     info = env.step(action)
    #     braininfo = info['Train_gail']
    #     dones_GailEnvPerson.append(info['GailEnvPerson'].local_done)
    #     dones_Train_gail.append(info['Train_gail'].local_done)
    # else:
    #     action = {'Train_gail': np.zeros([1,2])}
    #     info = env.step(action)
    #     braininfo = info['Train_gail']
    #     dones_GailEnvPerson.append(info['GailEnvPerson'].local_done)
    #     dones_Train_gail.append(info['Train_gail'].local_done)

    # info = env.step()
    # braininfo = info['Train_gail']
    # dones.append(braininfo.local_done)
    # if braininfo.agents==[]:
    #     print("step", i)
    # if braininfo.vector_observations.shape[0] == 0:
    #     print('err')
    # obs.append(braininfo.vector_observations)
# print(obs)
endtime = datetime.datetime.now()
print('time comsumed: ', (endtime - starttime).seconds)
print(dones_Train_gail[:99])
print(dones_Train_gail.__len__())
