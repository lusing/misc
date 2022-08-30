import gym

from stable_baselines3 import PPO

#env = gym.make('CartPole-v1')
#env = gym.make('ALE/Pong-v5',render_mode='human')
env = gym.make('Breakout-v0',render_mode='human')

#model = PPO('MlpPolicy', env, verbose=1)
model = PPO.load("Breakout-v0")
#model.learn(total_timesteps=10000)
#model.save("ppo_pong")

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