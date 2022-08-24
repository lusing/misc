import gym

def action(status): 
    pos, v, ang, va = status
    print(status)
    if pos <= 0: 
        return 1
    else: 
        return 0 

env = gym.make('CartPole-v0')
status = env.reset()
for step in range(1000):
    i = 0
    env.render()
    status, reward, done, info = env.step(action(status))
    if done: 
        print('dead in %d steps' % step)
        break
env.close()
