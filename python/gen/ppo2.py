import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    env = gym.make('PongNoFrameskip-v4')
    env = AtariWrapper(env)
    return env

# 创建Atari游戏环境，例如Pong
#env = make_env()

# 如果需要并行环境，可以使用VecEnv，例如：DummyVecEnv或SubprocVecEnv
#env = DummyVecEnv([lambda: env])

# 选择一个算法，例如PPO，并设置其参数
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=2.5e-4,
    n_steps=128,
    batch_size=256,
    n_epochs=4,
    gamma=0.99,
    ent_coef=0.01,
    clip_range=0.1,
    clip_range_vf=1,
)

# 训练模型
model.learn(total_timesteps=10000)

# 保存模型
model.save("ppo_pong")

# 加载模型并在环境中运行
model = PPO.load("ppo_pong")
env = DummyVecEnv([make_env])
obs = env.reset()
terminated = False
while not terminated:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
