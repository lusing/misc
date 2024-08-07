import gym

from stable_baselines3 import A2C
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

#env = gym.make('CartPole-v1')
#env = gym.make('ALE/Blackjack-v5',render_mode='human')
#env = gym.make('ALE/Pong-v5',render_mode='rgb_array')
env = gym.make('ALE/Breakout-v5',render_mode='rgb_array')
#env = gym.make('ALE/Breakout-v5',render_mode='human')

model = A2C('CnnPolicy', env, verbose=1, policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5)),vf_coef=0.25)
#model = A2C.load("a2c_pong")
#model.train()
model.learn(total_timesteps=50000)
model.save("a2c_breakout_v3")

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
