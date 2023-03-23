# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:39:29 2023

@author: a.aravindh
"""

from collections import deque
import random

# A cyclic buffer of bounded size that holds the transitions observed recently
class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def remember(self, *args):
        self.memory.append(*args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)