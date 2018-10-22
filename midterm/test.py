import gym 
import tensorflow
#import box2d


env = gym.make('BipedalWalker-v2')

env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
