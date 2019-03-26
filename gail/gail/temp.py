#!/usr/bin/env python
# encoding: utf-8
'''
@author: Huang Junfu
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 2504598262@qq.com
@software: 
@file: temp.py
@time: 2019/3/1 20:29
@desc:
'''
import logging


def constfn(val):
    def f(_):
        return val
    return f


def learn(lr):
    if isinstance(lr, float):
        lr = constfn(lr)
    else:
        assert callable(lr), "lr is not callable"

    for i in range(1, 10):
        frac = 1.0-i/10
        print(frac)
        lrnow = lr(frac)
        print(lrnow)


learn(1.0)