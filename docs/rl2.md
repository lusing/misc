# 强化学习环境升级 - 从gym到Gymnasium

作为强化学习最常用的工具，gym一直在不停地升级和折腾，比如gym[atari]变成需要要安装接受协议的包啦，atari环境不支持Windows环境啦之类的，另外比较大的变化就是2021年接口从gym库变成了gymnasium库。让大量的讲强化学习的书中介绍环境的部分变得需要跟进升级了。

不过，不管如何变，gym[nasium]作为强化学习的代理库的总的设计思想没有变化，变的都是接口的细节。

![](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/gymnasium.png)

## step和观察结果

总体来说，对于gymnasium我们只需要做两件事情：一个是初始化环境，另一个就是通过step函数不停地给环境做输入，然后观察对应的结果。

初始化环境分为两步。
第一步是创建gymnasium工厂中所支持的子环境，比如我们使用经典的让一个杆子不倒的CartPole环境：
```python
import gymnasium as gym
env = gym.make("CartPole-v1")
```
第二步，我们就可以通过env的reset函数来进行环境的初始化：
```python
observation, info = env.reset(seed=42)
```

我们可以将observation打印出来，它一个4元组，4个数值分别表示：
- 小车位置
- 小车速度
- 棍的倾斜角度
- 棍的角速度

如果角度大于12度，或者小车位置超出了2.4，就意味着失败了，直接结束。

小车的输入就是一个力，要么是向左的力，要么是向右的力。0是向左推小车，1是向右推小车。

下面我们让代码跑起来。

首先我们通过pip来安装gymnasium的包：
```
pip install gymnasium -U
```

安装成功之后，

```python
import gymnasium as gym
env = gym.make("CartPole-v1")

print(env.action_space)

observation, info = env.reset(seed=42)
steps = 0
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation)

    if terminated or truncated:
        print("Episode finished after {} steps".format(steps))
        observation, info = env.reset()
        steps = 0
    else:
        steps += 1
        
env.close()
```

env.action_space输出是Discrete(2)。也就是两个离散的值0和1。前面我们介绍了，这分别代表向左和向右推动小车。

observation输出的4元组，我们前面也讲过了，像这样：
[ 0.0273956  -0.00611216  0.03585979  0.0197368 ]

下面就是关键的step一步：
```python
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
```

刚才我们介绍了，CartPole的输入只有0和1两个值。我们采用随机让其左右动的方式来试图让小车不倒。

如果你觉得还是不容易懂的话，我们可以来个更无脑的，管它是什么情况，我们都一直往左推：
```python
observation, reward, terminated, truncated, info = env.step(0)
```
基本上几步就完了：
```
[ 0.02699083 -0.16518621 -0.00058549  0.3023946 ] 1.0 False False {}
[ 0.0236871  -0.36029983  0.0054624   0.5948928 ] 1.0 False False {}
[ 0.01648111 -0.5554978   0.01736026  0.88929135] 1.0 False False {}
[ 0.00537115 -0.750851    0.03514608  1.1873806 ] 1.0 False False {}
[-0.00964587 -0.94641054  0.0588937   1.4908696 ] 1.0 False False {}
[-0.02857408 -1.1421978   0.08871109  1.8013463 ] 1.0 False False {}
[-0.05141804 -1.3381925   0.12473802  2.1202288 ] 1.0 False False {}
[-0.07818189 -1.534317    0.16714258  2.4487078 ] 1.0 False False {}
[-0.10886823 -1.7304213   0.21611674  2.7876763 ] 1.0 True False {}
Episode finished after 8 steps
```
下面我们解释下返回的5元组，observation就是位置4元组，reward是用于强化学习的奖励，在本例中只要是不死就是1. terminated就是是否游戏结束了。
Truncated在官方定义中用于处理比如超时等特殊结束的情况。
truncated, info对于CartPole来说没有用到。

搭建好了gymnasium环境之后，我们就可以进行策略的升级与迭代了。
比如我们写死一个策略，如果位置小于0则向右推，反之则向左推：

```python
def action_pos(status): 
    pos, v, ang, va = status
    #print(status)
    if pos <= 0: 
        return 1
    else: 
        return 0 
```

或者我们根据角度来判断，如果角度大于0则左推，反之则右推：
```python
def action_angle(status): 
    pos, v, ang, va = status
    #print(status)
    if ang > 0: 
        return 1
    else: 
        return 0
```

角度策略的完整代码如下：
```python
import gymnasium as gym
env = gym.make("CartPole-v1")
#env = gym.make("CartPole-v1",render_mode="human")

print(env.action_space)
#print(env.get_action_meanings())

observation, info = env.reset(seed=42)
print(observation,info)

def action_pos(status): 
    pos, v, ang, va = status
    #print(status)
    if pos <= 0: 
        return 1
    else: 
        return 0 

def action_angle(status): 
    pos, v, ang, va = status
    #print(status)
    if ang > 0: 
        return 1
    else: 
        return 0

steps = 0
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action_angle(observation))
    print(observation, reward, terminated, truncated, info)

    if terminated or truncated:
        print("Episode finished after {} steps".format(steps))
        observation, info = env.reset()
        steps = 0
    else:
        steps += 1
        
env.close()
```

## 与老gym的主要区别

目前版本与之前gym的最主要区别在于step返回值从原来的4元组变成了5元组。
原来是observation, reward, done, info，而现在done变成了 terminated增加了truncated。

老版本的：
```python
 status, reward, done, info = env.step(0)
```

新版的：
```python
observation, reward, terminated, truncated, info = env.step(0)
```

因而，原来处理done的地方需要改成terminated或truncated:

```python
if terminated or truncated:
    something()
```

另外，env.reset函数目前返回的是两个值，而不是原来的一个值：

```python
obs,info = env.reset()
```

## Atari游戏

![](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/Atari.png)

我们通过gymnasium[atari]包来安装atari游戏的gymnasium支持。

```
pip install gymnasium[atari]
```

与之前的gym一样，gymnasium默认是不安装atari游戏的，需要通过accept-rom-license包来安装游戏。

```
!pip install gymnasium[accept-rom-license]
```

### 通过get_action_meanings来获取游戏支持的操作

之前的CartPole只知道是离散的两个值。而Atari游戏则可支持获取游戏支持的操作的含义：
```
['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
```

### rendor_mode

另外，针对于Atari游戏，render_mode现在是必选项了。要指定是显示成人类可看的human模式，还是只输出rgb_array的模式。

### 完整例子

我们以乒乓球游戏为例，组装让其运行起来：

```
import gymnasium as gym
env = gym.make("ALE/Pong-v5", render_mode="human")
observation, info = env.reset()

print(env.get_action_meanings())

scores = 0

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    #print(observation, reward, terminated, truncated, info)

    if terminated or truncated:
        print("Episode finished after {} steps".format(scores))
        observation, info = env.reset()
        scores = 0
    else:
        scores +=1

env.close()
```

完整的游戏支持列表可以在https://gymnasium.farama.org/environments/atari/ 官方文档中查到。

## gymnasium与强化学习算法库的结合

stable-baselines3等强化学习库已经对gymnasium进行了支持，所以我们可以在stable-baselines3中直接使用gymnasium的环境。

先安装库：
```
pip install gymnasium[atari]
pip install gymnasium[accept-rom-license]
pip install stable_baselines3
```

比如，我们用DQN算法来训练乒乓球游戏：

```python
import gymnasium as gym
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.dqn import CnnPolicy


game = 'ALE/Pong-v5'

env = gym.make(game,render_mode="rgb_array")

save_file = 'dqn_'+game;

print(env.action_space)
model = DQN(CnnPolicy, env, verbose=1,exploration_final_eps=0.01,exploration_fraction=0.1,gradient_steps=1,learning_rate=0.0001,buffer_size=10000)
model.set_env(env)
model.learn(total_timesteps=1000000, log_interval=10)
model.save(save_file)

obs,info = env.reset()

score = 0
rewards_sum = 0

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    score = score + 1
    rewards_sum += reward
    if reward > 0:
        print('win!!!', reward)

    if terminated or truncated:
        # obs = env.reset()
        print('finished', score)
        print('reward sum=', rewards_sum)
        break
```

上面的代码我们还可以做两处改进：
1. 如果存在save_file，我们可以直接加载模型，在原有模型上继续训练。
2. 我们可以增加一个测试模式，观看训练后模型打游戏的真实效果。

同时，在colab上运行的话，我们可以将模型保存到google drive上，这样可以避免每次重新训练。

先要挂载google drive：

```python
from google.colab import drive
drive.mount('/content/drive')
```

然后我们可以把模型保存到google drive上：

```python
import gymnasium as gym
import numpy as np

import time
from datetime import datetime

from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.dqn import CnnPolicy

game = 'ALE/Pong-v5'

#eval = True
eval = False

#cont = True
cont = False

print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

start_time = time.time()
start_date = datetime.now()

if eval:
    env = gym.make(game,render_mode="human")
else:
    env = gym.make(game,render_mode="rgb_array")

save_file = '/content/drive/MyDrive/rl/dqn_'+game;

print(env.action_space)

if eval:
    model = DQN.load(save_file)
    model.set_env(env)
else:
    if cont:
        model = DQN.load(save_file)
    else:
        model = DQN(CnnPolicy, env, verbose=1,exploration_final_eps=0.01,exploration_fraction=0.1,gradient_steps=1,learning_rate=0.0001,buffer_size=10000)

    model.set_env(env)
    model.learn(total_timesteps=1000000, log_interval=10)
    model.save(save_file)

obs,info = env.reset()

score = 0
rewards_sum = 0

while True:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if eval:
        env.render()

    score = score + 1
    rewards_sum += reward
    if reward > 0:
        print('win!!!', reward)

    if terminated or truncated:
        print('finished', score)
        print('reward sum=', rewards_sum)
        break

duration = time.time() - start_time
print('duration=', duration)

time_cost = datetime.now() - start_date
print('time cost=', time_cost)
```

有了上面的框架之后，我们把DQN算法换成PPO算法，就可以让PPO算法来玩乒乓球游戏了。

```python
import gymnasium as gym
import numpy as np

import time
from datetime import datetime

from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.dqn import CnnPolicy

game = 'ALE/Pong-v5'

#eval = True
eval = False

cont = True
#cont = False

print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

start_time = time.time()
start_date = datetime.now()

if eval:
    env = gym.make(game,render_mode="human")
else:
    env = gym.make(game,render_mode="rgb_array")

save_file = '/content/drive/MyDrive/rl/dqn_'+game;

print(env.action_space)

if eval:
    model = PPO.load(save_file)
    model.set_env(env) 
else:
    if cont:
        model = PPO.load(save_file)
    else:
        model = PPO(MlpPolicy, env, verbose=1,learning_rate=2.5e-4,clip_range=0.1,vf_coef=0.5,ent_coef=0.01,n_steps=128)    
    model.set_env(env)
    model.learn(total_timesteps=1000000, log_interval=10)
    model.save(save_file)

obs,info = env.reset()

score = 0
rewards_sum = 0

while True:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if eval:
        env.render()

    score = score + 1
    rewards_sum += reward
    if reward > 0:
        print('win!!!', reward)

    if terminated or truncated:
        print('finished', score)
        print('reward sum=', rewards_sum)
        break

duration = time.time() - start_time
print('duration=', duration)

time_cost = datetime.now() - start_date
print('time cost=', time_cost)
```

## 视频输出 - 从Monitor到RecordVideo

有时候我们希望把游戏的视频输出出来，gym曾经使用Monitor来实现。现在gymnasium则改用RecordVideo来实现。

使用RecordVideo需要先安装moviepy库：
```
pip install moviepy
```

然后从gymnasium.wrappers包中引用RecordVideo：

```python
from gymnasium.wrappers import RecordVideo
```

human模式是没有办法输出视频的，所以我们需要把human模式改成rgb_array模式。然后我们指定RecordVideo的输出目录就可以了：

```python
env = gym.make(game,render_mode="rgb_array")
env = RecordVideo(env, './video')
```

输出默认是mp4格式，如果需要其他格式，比如我们在网页中要显示成gif格式，可以使用ffmpeg来转换：

```p
ffmpeg -i rl-video-episode-0.mp4 -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 output.gif
```

![](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/pong1.gif)

我们换个游戏：

![](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/pig1.gif)

如果某大漠老师强烈bs你使用gif，那么也可以转成apng格式：

```
ffmpeg -i rl-video-episode-0.mp4  output.apng
```

效果如下：

![](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/rl1.apng)

最后，我们来看一下能打赢电脑的吧：

![](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/pong-win.gif)
