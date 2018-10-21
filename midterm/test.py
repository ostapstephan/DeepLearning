import gym 
import tensorflow



env = gym.make('BipedalWalker-v2')

for i in range(20):
	observation = env.reset()
	for t in range(100):
		env.render()

		action = 0#env.action_space.sample()
		print(env.observation_space)
		observation, reward, done, info = env.step(action)
		'''
		if done:
			print("Episode finished after {} timesteps".format(t+1))
			break

		'''
