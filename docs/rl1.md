# 强化学习

## 表格法

### 时序差分法

### 异策略学习概述

## 近似求解法

### 值函数近似法

### 策略梯度法

#### Actor-Critic算法

#### PPO算法

PPO(Proximal Policy Optimization)算法也被称作近端优化策略算法，它借助异策略的核心思想实现经验回放，进而简化策略函数的训练过程。

异策略学习是基于重要性采样实现的，即通过对与原分布不同的另一个分布进行采样估计原分布的性质。

### 深度强化学习

#### DQN算法

Q-Learning算法的基本原理是在有限的状态和行动空间中，通过探索和更新状态-行动值表(Q表)中的状态-行动值(Q值)，从而计算出智能体行动的最佳策略。然而，现实强化学习问题往往具有很大的状态空间和行动空间。因此，使用值函数近似法代替传统表格求解法是强化学习实际应用的首选。

DQN算法的大体框架借鉴传统强化学习中的Q-Learning算法，并采用神经网络估计状态-行动值。在此基础上主要进行了如下三方面的修改。

1. 利用深度卷积神经网络逼近值函数

2. 设置独立的Fixed Q-target处理Q-Learning算法中的TD误差

3. 在训练强化学习算法的过程中采用经验回报机制

---

A2C

https://arxiv.org/pdf/1602.01783.pdf

ACER：SAMPLE EFFICIENT ACTOR-CRITIC WITH EXPERIENCE REPLAY

https://arxiv.org/pdf/1611.01224.pdf

ACKTR： Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation

https://arxiv.org/pdf/1708.05144.pdf

DDPG：Continuous control with deep reinforcement learning

https://arxiv.org/pdf/1509.02971.pdf

Generative Adversarial Imitation Learning (GAIL)

https://arxiv.org/pdf/1606.03476.pdf

PPO：Proximal Policy Optimization Algorithms

https://arxiv.org/pdf/1707.06347.pdf

TRPO: Trust Region Policy Optimization

https://arxiv.org/pdf/1502.05477.pdf

