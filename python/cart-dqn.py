import gym

from stable_baselines3 import DQN

env = gym.make("CartPole-v1")

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000, log_interval=4)
model.save("dqn_cartpole2")

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_cartpole2")

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
        print("dead in %d steps" % i)
        break
env.close()
