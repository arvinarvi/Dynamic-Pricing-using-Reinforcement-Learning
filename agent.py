# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 16:43:13 2023

@author: a.aravindh
"""

import random
import numpy as np

class Agent:
    
    def __init__(self, size):
        self.action_size = size
    
    def act(self, epsilon, qvalues):       
        sample = random.random()
        if sample > epsilon:
            return np.argmax(qvalues)
        else:
            return random.randrange(len(qvalues))