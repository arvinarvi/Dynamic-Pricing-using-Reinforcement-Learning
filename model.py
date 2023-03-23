# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:38:22 2023

@author: a.aravindh
"""


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import History
from keras.losses import Huber
from support import reshape
import numpy as np

class Model:
    def __init__(self, state_size, action_size):
        
        self.learning_rate = 0.001 #learning rate for deep network
        self.lr= 0.6 #Learning rate for q value updation
        self.gamma = 0.01
        self.model = self.load_model(state_size,action_size)
        self.history = History()
    
    def load_model(self, state_size, action_size):
        model = Sequential()
        model.add(Dense(128,activation='relu', input_dim=state_size))
        model.add(Dense(128,activation='relu'))
        model.add(Dense(128,activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),loss=Huber(delta=1.5))
        return model
    
    def train(self, batch_size, replay, state_size, action_size):
        mini_batch = replay.sample(batch_size)
        inputs = np.zeros((batch_size, state_size))
        outputs = np.zeros((batch_size, action_size))
        
        for index, items in enumerate(mini_batch):
            
            current_state, action, next_state, reward = items
            target_f = self.model.predict(reshape(current_state, (1,state_size)), verbose=0)
            if next_state is None:
                target_f[0][action] = reward
            else:
                target = reward + self.gamma*np.max(self.model.predict(reshape(next_state, (1,state_size)), verbose=0)[0])
                target_f[0][action] = self.lr*target_f[0][action] + (1-self.lr)*target 
          
            
        inputs[index] = reshape(current_state, state_size)
        outputs[index] = target_f
        
        self.model.fit(inputs,outputs, epochs=8, batch_size=32, verbose=0,callbacks=[self.history])
            
        
        