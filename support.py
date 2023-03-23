# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:38:04 2023

@author: a.aravindh
"""
import numpy as np


def reshape(x, shape):
    return x.reshape(shape)


def clip(x):
    return np.sqrt(x)


def plus(x):
    return 0 if x <0 else x


def minus(x):
    return -x if x < 0 else 0