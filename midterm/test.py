#!/bin/python3.6.7
# Ostap Voynarovskiy
# CGML Midterm
# October 21 2018
# Professor Curro
# ok so when installing the environment we had to make sure 
# that the install of gym was in the pyenv install just as an fyi to myself

# these guys helped me through DQN as they had implemented it for Cartpole
# https://keon.io/deep-q-learning/

import gym 
import time
import keras
import random
from gym import wrappers
import numpy as np
import pandas as pd
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 400

class DQN:
	def __init__(self, stateSize,actionSize):
		self.stateSize = stateSize
		self.df = 3 #discretizationfactor
		self.actionSizeDiscretized = self.df**actionSize #81
		self.memory = deque(maxlen=2000)
		self.gamma = .95 
		self.eps = 1 
		self.epsMin = 0.01 #1 percent random actions
		self.epsDecay = 0.995 
		self.learningRate = 0.001 #adam LR
		self.model = self.buildModel()
		self.aMatrix = self.genAMatrix()
	
	def buildModel(self):
		model = Sequential()
		model.add(Dense(32, input_dim=self.stateSize, activation="elu"))
		model.add(Dense(32, activation="elu"))
		model.add(Dense(self.actionSizeDiscretized)) #no activation this is a regression 
		model.compile(loss="mse",optimizer=keras.optimizers.Adam())
		print(model.summary())
		return model

	def remember(self, state, action, reward, nextState, done):
		self.memory.append((state, action, reward, nextState, done))

	def act(self, state):
		if np.random.uniform() < self.eps:
			action = env.action_space.sample()
			action = np.rint(action*1.47)
			print("action",action)
		else:
			a = self.model.predict(state)
			action = self.aMatrix[a]
		return action

	def replay(self, batchSize):
		batch = random.sample(self.memory,batchSize)
		for state,action,reward,nextState,done in batch:
			target = reward
			if not done: #add future discounted reward
				target += self.gamma * np.amax(self.model.predict(nextState)[0])
			target_f = self.model.predict(state)
			print("TARGET IS: ",target)
			print("TARGET_F IS: ",target_f)
			
			#figure out how to get a target from actions?????
			target_f[action] = target
			self.model.fit(state,target_f, epochs = 1, verbose = 0)
		if self.eps > self.epsMin:
			self.eps *= self.epsDecay
	
	def genAMatrix(self):
		i = [-1,0,1]
		mat = []
		for a in i:
			for b in i:
				for c in i:
					for d in i:
						mat.append([a,b,c,d])
		#plz dont kill us curro 
		#we're so desperate
		print(mat)
		return mat


	def daftPunk(self, rw, name): #read = true #just for saving and loading model weights
		#Write it, cut it, paste it, save it,
 		#Load it, check it, quick, rewrite it 
		if(rw):
			self.model.load_weights(name)
		else:
			self.model.save_weights(name)



if __name__ == "__main__":
	env = gym.make('BipedalWalker-v2')
	#env = gym.make("Breakout-ram-v0")
	stateSize = env.observation_space.shape[0] 
	print (stateSize)
	print (env.action_space)
	actionSize = 4 #env.action_space.n
	agent = DQN(stateSize,actionSize)

	batchSize = 32

	for ep in range(EPISODES):
		done = False
		state = env.reset()
		state = np.reshape(state,[1,stateSize])
		for time in range(1600):

			#change time to 2000 for hardcore mode 
			if ep%20 ==0:
				env.render()
			action = agent.act(state)
			nextState, reward, done, info = env.step(action)
			reward = reward if not done else -10
			nextState = np.reshape(nextState, [1,stateSize])
			agent.remember(state,action,reward,nextState,done)

			if done:
				print("Episode: {}/{}, score: {}, e:{:.2}".format(ep,EPISODES,time,agent.epsilon))
				break
			if len(agent.memory)>batchSize: 
				#create initial set of "training data"
				agent.replay(batchSize)

			state = nextState

	while not done:
		#env.render()
		cnt +=1
		action = env.action_space.sample()
		observation,reward,done,info= env.step(action)
	print( "Lasted ", cnt, "frames")		



