
import gym
import numpy as np

import time
from datetime import datetime 

from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.dqn import CnnPolicy


#game = 'ALE/Adventure-v5'
# game = 'Adventure-ram-v0' # 探险类
game = 'ALE/Pong-v5'
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
#game = 'ALE/ElevatorAction-v5' # 0
#game = 'ALE/Enduro-v5' # 赛车
#game = 'ALE/FishingDerby-v5' # 钓鱼
#game = 'ALE/Freeway-v5' # 过马路？
#game = 'ALE/Frostbite-v5' # 跳格子
#game = 'ALE/Gopher-v5' # 打地mouse
#game = 'ALE/Gravitar-v5' # shoot
#game = 'ALE/Hero-v5' # 过关类
#game = 'ALE/IceHockey-v5' # 球类
#game = 'ALE/Jamesbond-v5' # 横版射击, seems good
#game = 'ALE/JourneyEscape-v5' # 倒计时
#game = 'ALE/Kangaroo-v5' # not good
#game = 'ALE/Krull-v5' # not good
#game = 'ALE/KungFuMaster-v5' # 对打
#game = 'ALE/MontezumaRevenge-v5' # 上楼过关
#game = 'ALE/MsPacman-v5' # 吃金币
#game = 'ALE/NameThisGame-v5' # 水下射击
#game = 'ALE/Phoenix-v5' # 射击，try
#game = 'ALE/Pitfall-v5' # 过关, 倒计时
#game = 'ALE/Pooyan-v5' # 猪小弟, good
#game = 'ALE/PrivateEye-v5' # 倒计时
#game = 'ALE/Qbert-v5' # 跳格子
#game = 'ALE/Riverraid-v5' # 射击
#game = 'ALE/RoadRunner-v5' # 跑路
#game = 'ALE/Robotank-v5' # 高级射击
#game = 'ALE/Seaquest-v5' # 水下攻击 good
#game = 'ALE/Skiing-v5' # 滑雪
#game = 'ALE/Solaris-v5' # shoot
#game = 'ALE/StarGunner-v5' # 射击类
#game = 'ALE/Tennis-v5' # 网球
#game = 'ALE/TimePilot-v5' # 飞机射击类
#game = 'ALE/Tutankham-v5' # 探索
#game = 'ALE/UpNDown-v5' #赛车类
#game = 'ALE/Venture-v5' # 迷宫类
#game = 'ALE/VideoPinball-v5' # 弹珠台
#game = 'ALE/WizardOfWor-v5' # 迷宫射击
#game = 'ALE/YarsRevenge-v5' # 射击
#game = 'ALE/Zaxxon-v5' # 高级射击
#game = 'ALE/SpaceInvaders-v5'

#env = gym.make('Pong-v0')

eval = True
#eval = False

cont = True
#cont = False

print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

start_time = time.time()
start_date = datetime.now()

if eval:
    env = gym.make(game,render_mode="human")
else:
    env = gym.make(game,render_mode="rgb_array")

#save_file = 'dqn_pong';
save_file = 'dqn_'+game;

print(env.action_space)
print(env.get_action_meanings())

#model = DQN(MlpPolicy, env, verbose=1)
if eval:
    model = DQN.load(save_file)
    model.set_env(env) 
else:
    if cont:
        model = DQN.load(save_file)
    else:
        model = DQN(CnnPolicy, env, verbose=1,exploration_final_eps=0.01,exploration_fraction=0.1,gradient_steps=1,learning_rate=0.0001,buffer_size=10000)    
    model.set_env(env)
    model.learn(total_timesteps=500000, log_interval=10,eval_log_path='logs/'+save_file+'_eval')
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

duration = time.time() - start_time
print('duration=', duration)

time_cost = datetime.now() - start_date
print('time cost=', time_cost)
