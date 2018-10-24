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
		self.render = True
		self.loadModel = False

		self.stateSize = stateSize
		self.df = 5 #discretizationfactor
		self.actionSizeDiscretized = self.df**actionSize

		self.memory = deque(maxlen=2000)
		self.gamma = .85
		self.tau = 0.125
		self.epsilon = 1
		self.epsilonMin = 0.01 #1 percent random actions
		self.epsilonDecay = 0.995


		self.learningRate = 0.005 #adam LR
		self.batchSize = 32

		self.model = self.buildModel()
		self.target_model = self.buildModel()

		self.aMatrix = self.genAMatrix()

		if self.loadModel:
			self.daftPunk(1,'./bipedal.h5')

	def buildModel(self):
		model = Sequential()
		model.add(Dense(24, input_dim=self.stateSize, activation="relu"))
		model.add(Dense(48, activation="relu"))
		model.add(Dense(24, activation="relu"))
		model.add(Dense(self.actionSizeDiscretized)) #no activation this is a regression
		model.compile(loss="mse",optimizer=keras.optimizers.Adam())
		print(model.summary())
		return model

	def remember(self, state, action, reward, nextState, done):
		self.memory.append((state, action, reward, nextState, done))

	def act(self, state):
		if np.random.uniform() < self.epsilon:
			action = round(np.random.uniform(self.actionSizeDiscretized-1))
		else:
			action = self.model.predict(state).argmax()

		return action

	def replay(self):
		if len(self.memory)<self.batchSize:
			return

		batches = random.sample(self.memory, self.batchSize)

		for batch in batches:
			state, action, reward, nextState, done = batch

			target = self.target_model.predict(state)
			if done:
				target[0][action] = reward
			else:
				qFuture = max(self.target_model.predict(nextState)[0])
				target[0][action] = reward + qFuture*self.gamma #add discounted award

			self.model.fit(state,target,epochs=1,verbose=0)

		if self.epsilon > self.epsilonMin:
			self.epsilon *= self.epsilonDecay

	def genAMatrix(self):
		i = [-1,-0.5,0,0.5,1]
		mat = []
		for a in i:
			for b in i:
				for c in i:
					for d in i:
						mat.append([a,b,c,d])
		#plz dont kill us curro
		#we're so desperate
		return mat

	def target_train(self):
		weights = self.model.get_weights()
		target_weights = self.target_model.get_weights()
		for i in range(len(target_weights)):
			target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
		self.target_model.set_weights(target_weights)

	def daftPunk(self, rw, name): #read = true #just for saving and loading model weights
		#Write it, cut it, paste it, save it,
 		#Load it, check it, quick, rewrite it
		if(rw):
			self.model.load_weights(name)
		else:
			self.model.save_weights(name)



def main():
	env = gym.make('BipedalWalker-v2')
	stateSize = env.observation_space.shape[0]
	actionSize = 4

	agent = DQN(stateSize,actionSize)

	for ep in range(EPISODES):
		done = False
		state = env.reset()
		state = np.reshape(state,[1,stateSize])

		for time in range(1600):
			if agent.render:
				env.render()

			action = agent.act(state)

			nextState, reward, done, info = env.step(agent.aMatrix[action])
			nextState = np.reshape(nextState, [1,stateSize])
			reward = reward if not done else -10 #penialize for dying

			agent.remember(state,action,(time+reward),nextState,done)
			state = nextState

			print("info: ",info)
			agent.replay()
			agent.target_train()

			if done:
				print("Episode: {}/{}, score: {}, e:{:.2}".format(ep,EPISODES,time,agent.epsilon))
				break

	while not done:
		#env.render()
		cnt +=1
		action = env.action_space.sample()
		observation,reward,done,info= env.step(action)
	print( "Lasted ", cnt, "frames")

if __name__ == "__main__":
	main()
