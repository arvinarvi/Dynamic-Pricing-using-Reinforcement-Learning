# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:46:05 2023

@author: a.aravindh
"""


import numpy as np
from support import clip, plus, minus

class PriceModel:
    
    def __init__(self):    
        self.q_0 = 5000
        self.unit_cost = 100
        self.a_q = 300
        self.b_q = 100
        self.k = 10
        self.T = 20
        price_step = 10
        price_max = 500
        self.prices = np.arange(price_step,price_max,price_step)
        np.random.shuffle(self.prices)
    
        
    def demand(self, pt, pt_1):
        delta = pt - pt_1
        high = self.a_q*clip(plus(delta))
        low =  self.b_q*clip(minus(delta))
        x = self.q_0 - self.k*pt - high + low
        return plus(x)
    
    def profit(self, pt, pt_1):
        return self.demand(pt, pt_1)*(pt - self.unit_cost)
    
    
    def profit_total(self, p):
        return self.profit(p[0], p[0]) + sum(map(lambda t: self.profit(p[t], p[t-1]), range(len(p))))