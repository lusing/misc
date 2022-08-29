
import gym
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.dqn import CnnPolicy

#game = 'Breakout-v0'
# game = 'ALE/Breakout-v0'
#game = 'Adventure-v0'
# env = gym.make('SpaceInvaders-v0')
# env = gym.make('Adventure-v4')
# env = gym.make('AirRaid-v0')
# env = gym.make('AirRaid-v0')
# env = gym.make('SpaceInvaders-v4')
# env = gym.make('Alien-v0')
# env = gym.make('Amidar-v0')
# env = gym.make('Assault-v0')
# env = gym.make('Asterix-v0')
# env = gym.make('Asteroids-v0')
# env = gym.make('Atlantis-v0')
# env = gym.make('BankHeist-v0') # 10000
# env = gym.make('BattleZone-v0')
# env = gym.make('BeamRider-v0') # 10000 主动前进型的
# env = gym.make('Berzerk-v0')
# env = gym.make('Bowling-v0')
# env = gym.make('Boxing-v0')
# env = gym.make('Carnival-v0')
# env = gym.make('Centipede-v0')

# game = 'Adventure-ram-v0' # 探险类
game = 'ALE/Pong-v5'
# game = 'AirRaid-ram-v0'
# game = 'Alien-ram-v0' # 探险类
# game = 'Amidar-ram-v0'
# game = 'Assault-ram-v0' # 射击类
# game = 'Asterix-ram-v0'
# game = 'Asteroids-ram-v0'
# game = 'Atlantis-ram-v0'
# game = 'BankHeist-ram-v0'
# game = 'BattleZone-ram-v0'
# game = 'BeamRider-ram-v0'
# game = 'Berzerk-ram-v0'
# game = 'Bowling-ram-v0'
# game = 'Boxing-ram-v0'
# game = 'Breakout-ram-v0'
# game = 'Carnival-ram-v0'
# game = 'Centipede-ram-v0'
# game = 'ChopperCommand-ram-v0' # 射击类
# game = 'CrazyClimber-ram-v0' # 0
# game = 'Defender-ram-v0'
# game = 'DemonAttack-ram-v0' # 射击类
# game = 'DoubleDunk-ram-v0' # 篮球
# game = 'ElevatorAction-ram-v0' # 0
# game = 'Enduro-ram-v0'
# game = 'FishingDerby-ram-v0' # 钓鱼
# game = 'Freeway-ram-v0'
# game = 'Frostbite-ram-v0'
# game = 'Gopher-ram-v0'
# game = 'Gravitar-ram-v0' # 0
# game = 'Hero-ram-v0' # 0
# game = 'IceHockey-ram-v0' # 球类
# game = 'Jamesbond-ram-v0' #
# game = 'JourneyEscape-ram-v0'
# game = 'Kangaroo-ram-v0'
# game = 'Krull-ram-v0'
# game = 'KungFuMaster-ram-v0' # 对打
# game = 'MontezumaRevenge-ram-v0' # 上楼过关
# game = 'MsPacman-ram-v0'
# game = 'NameThisGame-ram-v0'
# game = 'Phoenix-ram-v0' # 射击
# game = 'Pitfall-ram-v0' # 过关
# game = 'Pooyan-ram-v0'
# game = 'PrivateEye-ram-v0'
# game = 'Qbert-ram-v0'
# game = 'Riverraid-ram-v0'
# game = 'RoadRunner-ram-v0'
# game = 'Robotank-ram-v0' # 高级射击
# game = 'Seaquest-ram-v0' # 水下攻击
# game = 'Skiing-ram-v0' # 滑雪
# game = 'Solaris-ram-v0'
# game = 'StarGunner-ram-v0'
# game = 'Tennis-ram-v0' # 网球
# game = 'TimePilot-ram-v0'
# game = 'Tutankham-ram-v0' # 探索
# game = 'UpNDown-ram-v0'
# game = 'Venture-ram-v0'
# game = 'VideoPinball-ram-v0' # 弹珠台
# game = 'WizardOfWor-ram-v0'
# game = 'YarsRevenge-ram-v0'
# game = 'Zaxxon-v0'
# game = 'Zaxxon-ram-v0'


#env = gym.make('Pong-v0')
#env = gym.make(game,render_mode="human")
env = gym.make(game,render_mode="rgb_array")

save_file = 'dqn_pong';
#save_file = 'dqn_'+game;

print(env.action_space)
print(env.get_action_meanings())

#model = DQN(MlpPolicy, env, verbose=1)
model = DQN(CnnPolicy, env, verbose=1,exploration_final_eps=0.01,exploration_fraction=0.1,gradient_steps=1,learning_rate=0.0001,buffer_size=10000,learning_starts=100000)
# model = DQN.load(save_file)
model.set_env(env)
model.learn(total_timesteps=150000, log_interval=10)
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
