import gym
env = gym.make('CartPole-v0')
status = env.reset()
for step in range(1000):
    i = 0
    env.render()
    status, reward, done, info = env.step( (i+1) % 2)
    if done: 
        print('dead in %d steps' % step)
        break
env.close()
