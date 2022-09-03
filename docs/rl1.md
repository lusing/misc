# 强化学习

## 表格法

### 时序差分法

### 异策略学习概述

## 近似求解法

### 值函数近似法

#### 值函数近似

值函数近似法是用一个带参数的函数$\hat v(s,w)$近似表示观测到的真实值$v(s)$。

值函数近似属于监督学习的范畴，因此值函数近似法也包括3个重要元素：模型、指标和算法。

### 策略梯度法

与前面通过学习状态-行动值函数选择行动a不同，策略梯度法通过学习一个参数化策略Parameterized Policy选择行动a，行动的选择不再依赖于值函数的取值。在某些情况下，值函数被用于学习策略参数。

#### 策略梯度

##### 基本概念

值函数近似法矣策略梯度法可类比机器学习中的监督学习，都是用一个带参数的函数$\hat y=f(x,w)$近似表示观测得到的真实值y。值函数近似法近似估算状态-行动值函数，将状态s和行动a输入一个带参数的函数$\hat q(s,a,w)$中，用来近似表示观测到的真实值$q(s,a)$。策略梯度法用一个带参数的函数$\pi(a|s,\theta)$近似表示观测到的真实值$\pi(a|s)$.

首先分析策略梯度法中需要优化的模型。策略梯度法通过学习一个策略分布增强行动选择的随机性，达到对行动空间进行探索的目的。
因此，可以定义一个策略函数
$\pi(a|s,\theta)=P{A_t=a|S_t=s,\theta_t=\theta}$。

其含义为，在时刻t状态s下采取行动a的概率。

其次，定义性能指标函数$J(\theta)$，对参数化策略的效果进行评估。在离散场景下，策略函数$\pi_\theta$的性能指标$J(\theta)$为每个交互序列episode中初始状态$s_0$的状态值函数$v_\pi(s_0)$，即
$J(\theta)\dot=v_{\pi_\theta}(s_0)$。

在连续场景下无起始状态的概念，我们使用另外两种计算方式：一种是根据当前环境在策略$\pi_\theta$影响下的状态分布$\mu_{\pi_\theta}$期望。

另一种则是对每一个可能状态s下采取的每一个行动a计算其单位时间奖励期望。

最后建立优化算法。我们以最大化性能指标函数$J(\theta)$为目标，其于随机梯度下降法，计算t时刻\theta的梯度，迭代更新得到t+1时刻的参数，进而找到对应的最优策略。

$\theta_{t+1}=\theta_t+\alpha \nabla J(\theta_t)$

##### 策略梯度定理

#### 蒙特卡洛策略梯度

蒙特卡洛策略梯度法，即REINFORCE算法，使用实际采样获得的长期回报G近似估计策略定理中未知的$q_{\pi_\theta}(s,a)$。之所以可以引入蒙特卡洛法，是因为可以通过实际采样获取多个完整的交互序列。

由策略梯度定理：
$\nabla J(\theta)=E_{\pi_\theta}(s,a)\nabla_\theta log\pi(a|s,\theta)$

$q_\pi(s,a)$的定义为采用策略$\pi$后在状态s下采用行动a获得的期望回报：

$q_\pi(s,a)=E[G_t|S_t=s,A_t=a]$

在实际进行策略梯度学习时，我们通过采样的形式获取足够多的样本进行梯度期望的计算。蒙特卡罗的思想就是用随机样本估算所需的期望值，所以有以下计算：

$
\nabla J(\theta)=\mathbb E_{\pi_\theta}[q_{\pi_\theta}\log \pi(a|s,\theta)]
=\mathbb E_{\pi_\theta}[G_t \nabla_{\theta}\log \pi(a|s,\theta)]
$

上式括号内的表达式作为一个可以被采样计算的量，它的期望值即实际梯度。
REINFORCE算法利用该机制实现随机梯度下降算法。

蒙特卡洛策略梯度算法：

输入：一个可微的策略带参函数$\pi(a|s,theta)$
1. 初始化策略参数$\theta\in\mathbb R^{d'}$
2. 初始化模拟总次数N

repeat
    按照策略$\pi$生成一个episode: $S_0,A_0,R_1,...,S_{T-1},A_{T-1},R_T$
    for 该episode中的每一个时刻t=0,1,...,T-1 do
        $G_t \leftarrow$ 计算时刻t后的总回报$\sum_{k=t+1}^{T}\gamma^{k-t-1}R_k$
        $\theta_{t+1}=\theta_t+\alpha G_t \log \pi(A_t|S_t,\theta_t)$
until i < N;

也就是说，当$G_t>0$时，在状态$S_t$采取行动$A_t$后会获取不错的总回报，于是增加策略参数$\pi_\theta$在S_t采取A_t的概率；反之，则减少行动$A_t$被采取的概率。

REINFORCE算法的训练过程如下：
1. 创建一个策略网络
2. 根据当前策略与环境互动完成一个交互序列
3. 收集状态-行动对的对数概率
4. 计算每个状态-行动对的衰减长期回报
5. 计算梯度更新公式
6. 更新策略网络

REINFORCE算法的梯度更新公式如下：
$\nabla_\theta J(\theta)=\frac{1}{N}\sum_{i=1}^{N}(\sum_{t=1}^{T}\nabla_{\theta}\log \pi_\theta(A_t^i|S_t^i))(\sum_{t'=t+1}^T\gamma^{t'-t-1}r_{t'}^i)$

总之，蒙特卡洛策略梯度算法在理论上拥有好的收敛性，并最终能使随机策略梯度下降算法收敛到一个局部最优点。然而，蒙特卡洛算法的估算过程中引入了高方差High Variance，因此在实际应用中普遍会比值函数方法的学习速度慢。

#### 带基线的REINFORCE算法

可以通过引入基线baseline机制减少蒙特卡洛算法带来的高方差收敛慢的问题。

基线机制的实现方式可以是一个关于状态s的函数，与其选择的行动a无关：

$\nabla J(\theta)=\mathbb E_{\pi_\theta}[(q_\pi(s,a)-b(s))\nabla_{\theta}\pi(a|s,\theta)]$

现在，重新对REINFORCE算法中的随机策略梯度下降算法进行描述：

$\theta_{t+1}\leftarrow \theta_t +\alpha(G_t-b(S_t))\nabla_\theta\log\pi(A_t|S_t,\theta_t)$ 

使用值函数$\hat v(S_t,w),w\in\mathbb\R^m$实现基线机制是一种常用手段。这里采用蒙特卡洛方法学习参数w，与REINFORCE算法保持一致。下面给出引入基线机制后的REINFORCE算法：

带基线机制的蒙特卡洛策略梯度算法：(REINFORCE with baseline)

输入：一个可微的策略函数$\pi_\theta(a|s)$，一个可微的值函数$\hat v(s_t,w)$

1. 初始化策略参数$\theta\in \mathbb R^{d'}$和值函数参数$w\in \mathbb{R}^m$
2. 初始化步长因子$\alpha^\theta>0,\beta^w>0$
3. 初始化模拟总次数N

repeat
    按照策略$\pi_\theta$生成一个Episode: $S_0,A_0,R_1,S_1,...,S_{T-1},A_{T-1},R_{T}$;
    for 该Episode中的每一个时刻t=0,1,...,T-1 do
        $G_t\leftarrow 计算时刻t后的长期回报\sum_{k=t+1}^{T}\gamma^{k-t-1}R_{k}$
        $\delta\leftarrow G_t - \hat v(S_t,w)$;
        $w_{t+1}\leftarrow w_t + \beta^w\delta\nabla_w \hat v(S_t,w_t)$;
        $\theta_{t+1}\leftarrow \theta_t + \alpha^\theta \log\pi(A_t|S_t,\theta_t)$;

#### Actor-Critic算法

如果同时进行策略函数学习和值函数近似，则统称为Actor-Critic Methods.
Actor-Critic方法也是一种策略学习加速收敛的方法，它可以根据评估策略的不同实现方式分为多个种类。
在Actor-Critic算法中，Actor指的是策略函数近似Policy Approximation模块，它负责环境互动中选择行动；而Critic指的是值函数近似Value Approximation模块，它负责去评价Actor所做的行动。

#### PPO算法

PPO(Proximal Policy Optimization)算法也被称作近端优化策略算法，它借助异策略的核心思想实现经验回放，进而简化策略函数的训练过程。

异策略学习是基于重要性采样实现的，即通过对与原分布不同的另一个分布进行采样估计原分布的性质。当把异策略应用到策略梯度学习中时，与环境互动产生训练数据的策略函数与被训练的策略函数拥有两套参数$\theta'$和$\theta$。首先，负责互动的策略$\pi_{\theta'}$在环境中采样交互序列数据，而被训练的策略$\pi_\theta$利用这些交互序列样本进行策略参数学习。
所以，结合重要性采样原理，有以下异策略学习推理过程成立。

$
\nabla J(\theta) = \mathbb E_{\pi_\theta}[(q_{\pi_\theta}(s,a)-v_w(s)\nabla\log\pi(a|s,\theta))]
= \mathbb E_{\pi_{\theta}}[A(s,a)\nabla \log\pi(a|s,\theta)]
= \mathbb E_{\pi_{\theta'}}[\frac{\pi_\theta(a|s,\theta)}{\pi_{\theta'}(a|s,\theta')}A(s,a)\nabla\log\pi(a|s,\theta))]
$

其中，A(s,a)为优势函数。值得关注的是，这里的优势函数值将由$\theta_{\pi'}$采样样本估算而得。

根据上式，和
$\nabla f(x) = f(x) \nabla \log f(x)$，将$\pi(a|s,\theta)$看作f(x),$\theta$看作x，可以反推出异策略学习的目标函数为$J(\theta)$。目标是通过最大化下式中的目标函数以移动电视行策略参数学习，进行获取局部最优解。
$J^{CPI}(\theta) = \mathbb E_{\pi_{\theta'}}[\frac{\pi_\theta(a|s,\theta)}{\pi_{\theta'}(a|s,\theta')}A(s,a)]$。

其中，CPI指的是保守策略迭代Conserative Policy Iteration。

使用异策略学习的前提条件是，与环境互动的策略函数$\pi_{\theta'}$与被训练的策略函数$\pi_\theta$的参数分布上不能有太大的差距。

PPO算法引入修剪式概率比Clipping probability ratio，限定$r(\theta)=\frac{\pi_\theta(a|s,\theta)}{\pi_{\theta'}(a|s,\theta')}$的取值在1附近。
下面给出PPO算法的核心表达式：
$J^{CPI}(\theta) = \mathbb E_{\pi_{\theta'}}[\min[r(\theta)\hat{A}_{\theta'},clip(r(\theta),1-\epsilon,1+\epsilon)\hat{A}]]$。

其中，$\epsilon$为超参数，一般令$\epsilon=0.2$。$\hat{A}_{\theta'}$为基于策略函数$\pi_{\theta'}$采样样本计算而得的优势函数估计值。

下面给出基于小批量随机梯度下降Mini-batch Stochastic Gradient Descent的PPO算法的实现。

近端策略优化算法PPO:
for i=1,2,...,max_epochs do
    for 交互序列$epi_i$内的交互次数$j=1,2,...,max_steps$ do
        利用策略$\pi_\theta'$与环境进行互动;
        if j % update_timestep == 0 then
            计算优势函数估计值：$\hat{A}_{1},...,\hat{A}_{t},...,\hat{A}_{T},$;
            对目标函数$J^{CPI}(\theta)$进行K次优化求解：
        end if;
    end
end

### 深度强化学习

#### DQN算法

Q-Learning算法的基本原理是在有限的状态和行动空间中，通过探索和更新状态-行动值表(Q表)中的状态-行动值(Q值)，从而计算出智能体行动的最佳策略。然而，现实强化学习问题往往具有很大的状态空间和行动空间。因此，使用值函数近似法代替传统表格求解法是强化学习实际应用的首选。

DQN算法的大体框架借鉴传统强化学习中的Q-Learning算法，并采用神经网络估计状态-行动值。在此基础上主要进行了如下三方面的修改。

1. 利用深度卷积神经网络逼近值函数

当使用涳充卷积神经网络表示Q值时，针对高维连续状态空间与大规模行动空间的强化学习成为可能。然而，当实际使用神经网络表示Q值时，强化学习过程的后半部分会出现不稳定状态，进而不收敛。具体来说，学习效果刚开始是非常好的，智能体在与环境与动中表现得越来越好。但随着学习进程的推进，即使将步长因子$\alpha$设置得很小的数值，智能体很大概率也会做出糟糕的决定。这种时好时坏的过程会不断循环重复，进而难以实现学习收敛。
这种不稳定的原因有以下几个：
- 前后相邻的样本状态高度相关
- 不同于Q-Learning中每个步骤对状态-动作值的精确更新，在DQN中，每个网络参数的单步更新都可能引起策略分布的巨大变化，进而导致训练样本分布的巨大变化。
- 神经网络很容易出现过拟合，很难产生反映出全局环境信息的交互序列数据。
对此，我们采用“双网络”机制和经验回放机制Exprience Replay来帮助缓解上述问题。

2. 设置独立的Fixed Q-target处理Q-Learning算法中的TD误差

$Q(s,a)\leftarrow Q(s,a)+\alpha\left[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\right]$

如上式所示，表格求解法中Q-Learning算法，其在TD误差更新规则的推动下，每次基于单个样本(s,a,r,s')更新Q值时都会抹去原来的数据值。正如前面提出的，使用神经网络逼近的Q值对于每个参数的更新都是敏感的，每次网络参数迭代都会造成所有Q值变动，其中包括用于计算TD目标的目标Q值。这样一来，一直处于变动的目标值会影响网络训练的收敛性。因此，DQN算法所用的方法与监督学习中用到的方法相似，通过引入“双网络”机制减少目标Q值的变动。“双网络”包含用于参数训练的Q在线网络Online network和进行前向传播以生成目标Q值的Q目标网络Target Network，以目标Q值作为监督学习中的训练标签Label.

3. 在训练强化学习算法的过程中采用经验回报机制

DQN算法会将一段时间的数据作为一个批次进行集中训练，这一批数据集合称为经验Experience，而这与Q-Learning算法每次使用单个样本进行学习的过程有所不同。经验回放过程具体是指专门使用一块内存区域D存储一段时间内的(s,a,r,s')样本集，然后对该样本集做进一步的随机采样，进而得到一个用于值函数网络参数训练的小批量Mini-Batch的样本集。随机采样的过程打破了相邻样本的高度相关性，进一步提高了强化学习的稳定性。

#### DDPG算法

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

