import gym

from stable_baselines3 import A2C

#env = gym.make('CartPole-v1')
#env = gym.make('ALE/Blackjack-v5',render_mode='human')
env = gym.make('ALE/Pong-v5',render_mode='rgb_array')

model = A2C('MlpPolicy', env, verbose=1)
#model = A2C.load("a2c_pong")
#model.train()
model.learn(total_timesteps=10000)
model.save("a2c_pong")

model = A2C.load("a2c_pong")

model.learn(total_timesteps=10000)

model.save("a2c_pong")

obs = env.reset()

score = 0
rewards_sum = 0

while True:
    # print(score)
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    #env.render()
    score = score + 1
    rewards_sum += reward
    if reward > 0:
        print('win!!!', reward)

    if done:
        # obs = env.reset()
        print('finished', score)
        print('reward sum=', rewards_sum)
        break
env.close()
