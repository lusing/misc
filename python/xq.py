import gym
import time

env = gym.make('gym_xiangqi:xiangqi-v0')

env.reset()

print(env.action_space)

done = False
while not done:
    #time.sleep(1)
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
env.close()