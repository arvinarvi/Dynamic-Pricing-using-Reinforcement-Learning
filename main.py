# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 10:40:03 2023

@author: a.aravindh
"""

from env import Environment
from agent import Agent
from model import Model
from greedyeps import AnnealedEpsGreedyPolicy
from memory import ReplayMemory
from support import reshape
from pricemodel import PriceModel
import numpy as np
from utility import plot_price_schedules, plot_return_trace

EPISODE = 5000
BATCH_SIZE=32

pmodel = PriceModel()
env = Environment(pmodel.T)
action_size = len(pmodel.prices)
agent = Agent(action_size)
replay = ReplayMemory(2000)
policy = AnnealedEpsGreedyPolicy()
qm = Model(env.state_size, agent.action_size)

return_trace = []
profit_trace = []
T = pmodel.T
for i in range(EPISODE):
    
    reward_trace = []
    p = []
    env.reset_state(T)
     #get the initialized current state
    for t in range(T):
         
        current_state = env.state
        qvalues = qm.model.predict(reshape(current_state, (1,env.state_size)), verbose=0)[0]#give the current state to the Deep Q network  and get the action based on q value
        eps = policy.decayeps()
        action = agent.act(eps, qvalues)
        env.step(t, action, pmodel)
        next_state = env.state#compute the next state from action 
        env.compute_reward(pmodel)
        reward = env.reward#compute the reward based on the action
        replay.remember((current_state, action, next_state, reward))#record the state
 
        reward_trace.append(reward)
        p.append(pmodel.prices[action])
        
    #train the model
    if len(replay) > BATCH_SIZE:
        qm.train(BATCH_SIZE, replay, env.state_size, agent.action_size)
        print(f"Episode: {i+1}/{EPISODE}, e:{eps},Val_loss:{np.mean(qm.history.history['loss'])}")
        
    return_trace.append(sum(reward_trace))
    profit_trace.append(p)


plot_return_trace(return_trace)
plot_price_schedules(profit_trace, T, 5, 1)

for profit in sorted(pmodel.profit_total(s) for s in profit_trace)[-10:]:
    print(f'Best profit results: {profit}')