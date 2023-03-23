# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:33:46 2023

@author: a.aravindh
"""

import numpy as np

class Environment:
    
    def __init__(self, T):
        #self.state = prices[np.random.randint(0,20,20)]
        #enc_timestamp = np.zeros(10)
        #np.append(self.state,enc_timestamp)
        self.state = np.zeros(2*T)
        self.reward = 0
        self.state_size = len(self.state)
        
    def step(self, t, action, pmodel):
        T = pmodel.T
        if t == T-1: #Already in last state
            return None
        next_state = np.repeat(0, len(self.state))
        next_state[0] = pmodel.prices[action]
        next_state[1:T] = self.state[0:T-1]
        next_state[t+T] = 1
        self.state = next_state
        
    def compute_reward(self, pmodel):
        self.reward = pmodel.profit(self.state[0], self.state[1])
        
    def reset_state(self, T):
        self.state = np.zeros(2*T)
        self.reward = 0