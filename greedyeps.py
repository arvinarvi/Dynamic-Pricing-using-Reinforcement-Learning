# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:10:19 2023

@author: a.aravindh
"""

import math

class AnnealedEpsGreedyPolicy:
    def __init__(self, eps_start = 0.9, eps_end = 0.05, eps_decay = 400):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

    def decayeps(self):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        return eps_threshold