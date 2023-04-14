# 导入必要的库
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

test = False # 修改为True以训练模型
test = True # 修改为True以训练模型

# 创建一个Atari游戏环境，例如Pong
#env = gym.make("PongNoFrameskip-v4",render_mode="human")

env = make_vec_env("PongNoFrameskip-v4",n_envs=4)

# 应用Atari预处理包装器
#env = AtariWrapper(env)

# 如果需要并行环境，可以使用VecEnv，例如：DummyVecEnv或SubprocVecEnv
#env = DummyVecEnv([lambda: env])

if test:

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
    model.learn(total_timesteps=100000)

# 保存模型
    model.save("ppo_pong")

env = gym.make("PongNoFrameskip-v4",render_mode="human")

# 加载模型并在环境中运行
model = PPO.load("ppo_pong")
done = False
obs, states = env.reset()
terminated = False # 修改done为terminated
while not done: # 修改循环条件为terminated
    action, _states = model.predict(obs)
    obs, reward, done ,done2, info = env.step(action) # 修改返回值为terminated和truncated
    env.render()
    #print(reward)
