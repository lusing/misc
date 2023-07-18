import gym


game = 'ALE/Pooyan-v5'
game = 'ALE/Breakout-v5'

# env = gym.make(game,render_mode='human')
# env = gym.make(game,render_mode='human')
env = gym.make(game)

print(env.action_space)
print(env.get_action_meanings())

env.reset()

for _ in range(100):
    #env.render()
    a = env.action_space.sample()
    print(a)
    env.step(a) # take a random action
env.close()
