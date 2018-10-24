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
import pylab
from gym import wrappers
import numpy as np
import pandas as pd
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 4000000

class DDQN:
	def __init__(self, stateSize,actionSize):
		self.render = True
		self.loadModel = False

		self.stateSize = stateSize
		self.actionSize = actionSize

		self.memory = deque(maxlen=10000000)
		self.gamma = .99
		self.tau = 0.125
		self.epsilon = 1.0
		self.epsilonMin = 0.01 #1 percent random actions
		annealedFrames = 250000
		self.epsilonDecay = (self.epsilon-self.epsilonMin)/annealedFrames

		self.learningRate = 0.0000625
		self.adamEpsilon = 0.00015

		self.trainingStart = 200000
		self.batchSize = 64

		self.model = self.buildModel()
		self.targetModel = self.buildModel()

		#if self.loadModel:
		#	self.daftPunk(1,'./luka_1540391782.5540128')
		#	self.epsilon = self.epsilonMin

		self.updateTargetModel()

	def buildModel(self):
		model = Sequential()
		model.add(Dense(128, input_dim=self.stateSize, activation="relu"))
		model.add(Dense(32, activation="relu"))
		model.add(Dense(self.actionSize, activation="linear")) #no activation this is a regression
		model.compile(loss="mse",optimizer=keras.optimizers.Adam(lr=self.learningRate, epsilon=self.adamEpsilon))
		#print(model.summary())
		return model

	def remember(self, state, action, reward, nextState, done):
		self.memory.append((state, action, reward, nextState, done))

	def act(self, state):
		if np.random.uniform() < self.epsilon:
			return random.randrange(self.actionSize)
		else:
			return self.model.predict(state).argmax()

	def replay(self):
		if len(self.memory)<self.trainingStart:
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
			self.epsilon -= self.epsilonDecay

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
			name = "breakout_gpu" + str(time.time())
		if rw:
			self.model.load_weights(name)
			print("load success")
		else:
			self.model.save_weights(name)


def main():
	env = gym.make('Breakout-ram-v0')
	stateSize = env.observation_space.shape[0]
	actionSize = env.action_space.n

	agent = DDQN(stateSize,actionSize)

	scores, episodes = [], []

	for ep in range(EPISODES):
		done = False
		state = env.reset()
		state = np.reshape(state,[1,stateSize])

		score = 0
		lives = 5

		while not done:
			if ep % 20 == 0:
				env.render()

			action = agent.act(state)

			nextState, reward, done, info = env.step(action)
			nextState = np.reshape(nextState, [1,stateSize])
			#reward = reward if not done else -10 #penialize for dying

			agent.remember(state,action,reward,nextState,done)
			agent.replay()

			state = nextState
			score += reward
			dead = info['ale.lives']<lives
			lives = info['ale.lives']
			# if an action make the Pacman dead, then gives penalty of -100
			reward = reward if not dead else -100


		if done:
			scores.append(score)
			episodes.append(ep)

			if ep%25 == 1:
				pylab.plot(episodes, scores, 'b')
				pylab.xlabel('Episodes')
				pylab.ylabel('Score')
				pylab.title('Breakout: Episodes vs Score')
				pylab.savefig("./breakout.pdf")

				agent.daftPunk(0)

			print("Episode: {}/{}, reward: {}, e:{:.2}".format(ep,EPISODES,reward,agent.epsilon))


if __name__ == "__main__":
	main()
