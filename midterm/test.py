#!/bin/python3.6.7
# Ostap Voynarovskiy
# CGML Midterm
# October 21 2018
# Professor Curro
# ok so when installing the environment we had to make sure 
# that the install of gym was in the pyenv install just as an fyi to myself
import gym 
import time
import keras
from gym import wrappers
import numpy as np
import pandas as pd
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQN:
	def __init__(self, stateSize,actionSize):
		self.stateSize = stateSize
		self.actionSize = actionSize
		self.memory = deque(maxlen=2000)
		self.gamma = .95 
		self.epsilon = 1 
		self.epsilonMin = 0.01 #1 percent random actions
		self.epsilonDecay = 0.995 
		self.learningRate = 0.001 #adam LR
		self.model = self.buildModel()
	
	def buildModel(self):
		model = Sequential()
		model.add(Dense(32, input_dim=self.stateSize, activation="elu"))
		model.add(Dense(32, activation="elu"))
		model.add(Dense(self.actionSize )) #no activation this is a regression 
		model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam())
		print(model.summary())

	def 




if __name__ == "__main__":
	env = gym.make('BipedalWalker-v2')
	env.reset()
	done = False
	while not done:
		#env.render()
		cnt +=1
		action = env.action_space.sample()
		observation,reward,done,info= env.step(action)
	print( "Lasted ", cnt, "frames")		



