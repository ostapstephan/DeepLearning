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
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

EPISODES = 40000

class DQN:
	def __init__(self, stateSize,actionSize):
		self.stateSize = stateSize
		self.df = 3 #discretizationfactor
		self.actionSizeDiscretized = self.df**actionSize # 81
		self.memory = deque(maxlen=200000)
		self.gamma = .95 
		self.eps = 1 
		self.epsMin = 0.015 #1 percent random actions
		self.epsDecay = 0.9999
		self.learningRate = 0.001 #adam LR
		self.model = self.buildModel()
		self.aMatrix = self.genAMatrix()
	
	def buildModel(self):
		model = Sequential()
		model.add(Dense(64, input_dim=self.stateSize, activation="elu"))
		model.add(Dropout(.3))
		model.add(Dense(128, activation="elu"))
		model.add(Dropout(.3))
		model.add(Dense(self.actionSizeDiscretized)) 
		model.compile(loss="mse",optimizer=keras.optimizers.Adam()) # no activation this is a regression 
		print(model.summary())
		return model

	def remember(self, state, action, reward, nextState, done):
		self.memory.append((state, action, reward, nextState, done))

	def act(self, state):
		if np.random.uniform() < self.eps:
			#action = env.action_space.sample()
			#action = np.rint(action*1.47)
			#print("action",action)
			action = round(np.random.uniform(0,80))
		else:
			a = self.model.predict(state)
			action = np.argmax(a)

		return int(action)

	def replay(self, batchSize):
		batch = random.sample(self.memory,batchSize)
		for state,action,reward,nextState,done in batch:
			target = reward
			if not done: #add future discounted reward
				#print("model",self.model.predict(nextState))
				target += self.gamma * np.amax(self.model.predict(nextState)[0])
			target_f = self.model.predict(state)
			target_f[0][action] = target
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
		#print(mat)
		return mat


	def daftPunk(self, rw, name=None): #read = true #just for saving and loading model weights
		#Write it, cut it, paste it, save it,
		#Load it, check it, quick, rewrite it 
		if name == None:
			name = "i" +str(time.time())
		if rw:
			self.model.load_weights(name)
			print("load success")
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
	agent
	batchSize = 32
	#agent.daftPunk(True,"i1540357040.3333735")
	for ep in range(EPISODES):
		done = False
		state = env.reset()
		state = np.reshape(state,[1,stateSize])
		if ep%25 == 1:
			agent.daftPunk(False)

		t1 =  time.time()
		for tim in range(1600):
			#change time to 2000 for hardcore mode 
			
			if ep%20 ==0:
				env.render()
			action = agent.act(state)
			nextState, reward, done, info = env.step(agent.aMatrix[action])
			reward = reward if not done else -10
			nextState = np.reshape(nextState, [1,stateSize])

			agent.remember(state,action,reward,nextState,done)

			if done:
				t2 = time.time()
				print("Episode: {}/{}, score: {}, e:{:.2}, time {},{}".format(ep,EPISODES,reward,agent.eps,tim,t2-t1))
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
		