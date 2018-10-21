import gym 
import tensorflow


env = gym.make('Humanoid-v2')

for i in range(20):
	observation = env.reset()
	for t in range(100):
		env.render()
		print(observation)
		print(env.action_space)
		action = 0#env.action_space.sample()
		print(env.observation_space)
		observation, reward, done, info = env.step(action)
		'''
		if done:
			print("Episode finished after {} timesteps".format(t+1))
			break
		'''
