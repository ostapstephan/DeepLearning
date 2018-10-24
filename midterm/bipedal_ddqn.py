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

EPISODES = 40000

class DDQN:
	def __init__(self, stateSize,actionSize):
		self.render = True
		self.loadModel = False

		self.stateSize = stateSize
		self.df = 3 #discretizationfactor
		self.actionSizeDiscretized = self.df**actionSize

		self.memory = deque(maxlen=900000)
		self.gamma = .95
		self.tau = 0.125
		self.epsilon = 1
		self.epsilonMin = 0.01 #1 percent random actions
		self.epsilonDecay = 0.9995


		self.learningRate = 0.005 #adam LR
		self.batchSize = 32

		self.model = self.buildModel()
		self.targetModel = self.buildModel()
		self.updateTargetModel()

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

			target = self.targetModel.predict(state)
			if done:
				target[0][action] = reward
			else:
				qFuture = np.argmax(self.model.predict(nextState)[0])
				qTarget = self.targetModel.predict(nextState)[0]
				target[0][action] = reward + self.gamma*qTarget[qFuture] #add discounted award

			self.model.fit(state,target,epochs=1,verbose=0)

		if self.epsilon > self.epsilonMin:
			self.epsilon *= self.epsilonDecay

	def genAMatrix(self):
		i = [-1,0,1] #[-1 + x*(1-(-1))/(self.actionSizeDiscretized-1) for x in range(self.actionSizeDiscretized)]
		mat = []
		for a in i:
			for b in i:
				for c in i:
					for d in i:
						mat.append([a,b,c,d])
		#plz dont kill us curro
		#we're so desperate
		return mat

	def updateTargetModel(self):
		weights = self.model.get_weights()
		targetWeights = self.targetModel.get_weights()
		for i in range(len(targetWeights)):
			targetWeights[i] = weights[i] * self.tau + targetWeights[i] * (1 - self.tau)

		self.targetModel.set_weights(targetWeights)


	def daftPunk(self, rw, name=None): #read = true #just for saving and loading model weights
		#Write it, cut it, paste it, save it,
		#Load it, check it, quick, rewrite it
		if name == None:
			name = "Luka" +str(time.time())
		if rw:
			self.model.load_weights(name)
			print("load success")
		else:
			self.model.save_weights(name)


def main():
	env = gym.make('BipedalWalker-v2')
	stateSize = env.observation_space.shape[0]
	actionSize = 4

	agent = DDQN(stateSize,actionSize)
	Tim = 200
	for ep in range(EPISODES):
		done = False
		state = env.reset()
		state = np.reshape(state,[1,stateSize])

		if Tim < 1600:
			Tim+=50

		if ep%25 == 0:
			agent.daftPunk(0)#save
		t1 = time.time()

		for tim in range(Tim):
			if ep %25 ==0: #render
				env.render()
			action = agent.act(state)

			nextState, reward, done, info = env.step(agent.aMatrix[action])
			nextState = np.reshape(nextState, [1,stateSize])
			#reward = reward if not done else -10 #penialize for dying

			agent.remember(state,action,reward,nextState,done)
			state = nextState

			agent.replay()

			if done:
				agent.updateTargetModel()
				t2 = time.time()
				print("Episode: {}/{}, score: {}, e:{:.2},time,{},{}".format(ep,EPISODES,reward,agent.epsilon,tim,t2-t1))
				break




if __name__ == "__main__":
	main()
