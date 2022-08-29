
import gym
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.dqn import CnnPolicy


#game = 'ALE/Adventure-v5'
# game = 'Adventure-ram-v0' # 探险类
#game = 'ALE/Pong-v5'
#game = 'ALE/AirRaid-v5' # 也是大密蜂类
# game = 'ALE/Alien-v5' # 探险类
#game = 'ALE/Amidar-v5' # 迷宫类
#game = 'ALE/Assault-v5' # 射击类
#game = 'ALE/Asterix-v5' # 上下类
#game = 'ALE/Asteroids-v5'
# game = 'ALE/Atlantis-v5' # 防空
#game = 'ALE/BankHeist-v5' # 迷宫
#game = 'ALE/BattleZone-v5' # 坦克 good
# game = 'ALE/BeamRider-v5' # 太空射击
# game = 'ALE/Berzerk-v5' # 迷宫射击
# game = 'ALE/Bowling-v5' # ball
# game = 'ALE/Boxing-v5' # 时间长
# game = 'ALE/Breakout-ram-v5'
# game = 'ALE/Carnival-v5' # 限时射击
# game = 'ALE/Centipede-v5' # shoot
##game = 'ALE/ChopperCommand-v5' # 射击类
#game = 'ALE/CrazyClimber-v5' # 0 向上爬
# game = 'ALE/Defender-v5' # 射击类
# game = 'ALE/DemonAttack-v5' # 射击类
# game = 'ALE/DoubleDunk-v5' # 篮球
game = 'ALE/ElevatorAction-v5' # 0
#game = 'ALE/Enduro-v5' # 赛车
# game = 'ALE/FishingDerby-ram-v5' # 钓鱼
# game = 'ALE/Freeway-ram-v5'
# game = 'ALE/Frostbite-ram-v5'
# game = 'ALE/Gopher-ram-v0'
# game = 'ALE/Gravitar-ram-v0' # 0
# game = 'ALE/Hero-ram-v0' # 0
# game = 'ALE/IceHockey-ram-v0' # 球类
# game = 'ALE/Jamesbond-ram-v0' #
# game = 'ALE/JourneyEscape-ram-v0'
# game = 'ALE/Kangaroo-ram-v0'
# game = 'ALE/Krull-ram-v0'
# game = 'ALE/KungFuMaster-ram-v0' # 对打
# game = 'ALE/MontezumaRevenge-ram-v0' # 上楼过关
# game = 'ALE/MsPacman-ram-v0'
# game = 'ALE/NameThisGame-ram-v0'
# game = 'ALE/Phoenix-ram-v0' # 射击
# game = 'ALE/Pitfall-ram-v0' # 过关
# game = 'ALE/Pooyan-ram-v0'
# game = 'ALE/PrivateEye-ram-v0'
# game = 'ALE/Qbert-ram-v0'
# game = 'ALE/Riverraid-ram-v0'
# game = 'ALE/RoadRunner-ram-v0'
# game = 'ALE/Robotank-ram-v0' # 高级射击
# game = 'ALE/Seaquest-ram-v0' # 水下攻击
# game = 'ALE/Skiing-ram-v0' # 滑雪
# game = 'ALE/Solaris-ram-v0'
# game = 'ALE/StarGunner-ram-v0'
#game = 'ALE/Tennis-v5' # 网球
# game = 'ALE/TimePilot-ram-v0'
# game = 'ALE/Tutankham-ram-v0' # 探索
# game = 'ALE/UpNDown-ram-v0'
# game = 'ALE/Venture-ram-v0'
game = 'ALE/VideoPinball-v5' # 弹珠台
#game = 'ALE/WizardOfWor-v5' # 迷宫射击
#game = 'ALE/YarsRevenge-v5' # 射击
#game = 'ALE/Zaxxon-v5' # 高级射击


#env = gym.make('Pong-v0')
env = gym.make(game,render_mode="human")
#env = gym.make(game,render_mode="rgb_array")

#save_file = 'dqn_pong';
save_file = 'dqn_'+game;

print(env.action_space)
print(env.get_action_meanings())

#model = DQN(MlpPolicy, env, verbose=1)
model = DQN(CnnPolicy, env, verbose=1,exploration_final_eps=0.01,exploration_fraction=0.1,gradient_steps=1,learning_rate=0.0001,buffer_size=10000)
# model = DQN.load(save_file)
model.set_env(env)
model.learn(total_timesteps=5000, log_interval=10)
model.save(save_file)

obs = env.reset()

score = 0
rewards_sum = 0

while True:
    # print(score)
    action, _states = model.predict(obs, deterministic=True)
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
