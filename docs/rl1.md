# 强化学习

## 表格法

### 蒙特卡洛法

#### 蒙特卡洛预测

### 时序差分法

#### TD(0)预测

#### TD(0)控制: Sarsa(0)算法

#### n步时序差分预测

#### n步时序差分控制: n步Sarsa算法

### 异策略学习概述

#### 重要性采样

#### 每次访问与异策略学习

#### 异策略蒙特卡洛控制

#### 异策略时序差分控制：Q-Learning

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

我们来具体看下DQN算法的求解步骤。根据$\epsilon-greedy$策略执行动作a，然后把一段时间的经验数据存储到内存$\mathcal D$中，再从D中随机抽取单个样本(s,a,r,s')。在DQN算法中，我们会建立两个结构一模一样的值函数近似网络，其中Q目标网络的参数$w^-$会在一次批量训练中进行固定，并用于生成目标Q值，以作为标签数据。Q在线网络则用来评估策略，其网络参数w在每次迭代中都会更新。采用均方误差计算Q网络训练的损失函数为
$L(w)=\mathbb E[(R+\gamma\max_{a'}Q(s',a';w-)-Q(s,a;w))^2]$

值得注意的是，Q目标网络的参数并不是一直不变的。在Q在线网络获得一定次数的更新后，其最新的网络权重参数会直接用于更新Q目标网络，以作为下一轮目标网络的固定参数，循环以往。

训练过程：
- 随机行动并存储记忆
- 随机抽取批量样本
- 计算误差
- 更新在线网络
- 更新目标网络

DQN算法 Expericence Replay

输入：目标网络$\hat Q(s,a;w^-)和在线网络Q(s,a;w)$
输出：在线网络Q(s,a;w)
1. 初始化用于存储经验数据的内存D，其容量大小为N;
2. 初始化在线网络参数w;
3. 初始化目标网络参数$w^-\leftarrow w$
for episode = 1 : M do
    初始化状态$S_1$
    for t=1 : T do
        按照$\epsilon-greedy$策略选择一个行动A
        智能体执行行动$A_t$并观察到奖励$R_t$，以及新的状态$S_{t+1}$
        将$(S_t,A_t,R_t,S_{t+1})$存储到内存D中
        从内存D中随机采样一个小样本集$(S_j,A_j,R_j,S_{j+1})$
        设置
        $
        y_j=
        \begin{cases}
            R_j, S_{j+1}是终止态；\\
            R_j + \gamma\max_{a'}\hat Q(S_{j+1},a';w^-), S_{j+1}不是终止态；
        \end{cases}
        $

        针对$(y_j-Q(S_j,A_j;w))^2$使用mini-batch梯度下降法更新参数w

        每C轮参数更新后重设$\hat Q\leftarrow Q$，即$w^- \leftarrow w$
return w

DQN算法的经验回放机制让智能体反复与环境进行互动，以此积累经验数据。直到数据存储到一定的量，如达到数量N，就开始从D中进行随机采样并进行小批次的梯度下降计算Mini-Batch Gradient Descent。值得注意的是，在DQN算法中，强化学习部分Q-Learning算法和深度学习部分的随机梯度下降法是同步进行的，其中通过Q-Learning算法获取无限量的训练样本，然后对神经网络进行梯度下降训练。

综上所述，DQN算法利用经验回放机制增加了数据的利用率，同时也打破了经验数据之间的相关性，从而降低了模型参数方差，避免了过拟合。除此之外，DQN算法通过设定一个固定Q目标网络，解决了使用神经网络作为近似函数训练不收敛的问题。

#### DDPG算法

DDPG Deep Deterministic Policy Gradient算法引入了DQN的经验回放和固定目标网络这两个技巧来延续非线性值函数近似学习的稳定性和鲁棒性，并与策略梯度法中最简单的Actor-Critic算法结构相结合，旨在解决连续高维动作空间下的强化学习问题。

相对于策略梯度算法中使用随机性策略以确保探索的可能性，确定性策略的DDPG算法则通过异策略机制确保智能体能探索到潜在高回报动作，即根据随机策略$\mu'$(通过Ornstein-Uhlenbeck过程添噪声样本到确定性策略$\mu$上实现随机策略)选择行动以确保足够的探索，然后学习一个确定性策略$\mu$。

下面给出DDPG算法的过程：
- 基于确定性策略$\mu$的状态-行动值函数$Q^{\mu}(S_t,A_t)$
$Q^{\mu}(S_t,A_t) = \mathbb E[r(S_t,A_t) + \gamma Q^{\mu}(S_{t+1},\mu(S_{t+1}))]$
- 通过对Actor遵循的确定性策略$\mu$添加来自Ornstein-Uhlenbeck过程N的噪声样本，实现探索随机$\mu'$，其中$\mu'(S_t|\theta_t^\mu)$为关于参数$\theta_t^\mu$的策略近似函数网络。
- 针对Critic在线网络进行参数$\theta_t^Q$的链式求导，并基于策略梯度定理计算$\nabla_{\theta^\mu} J(\theta_t^\mu)$以用于学习Actor确定性策略。

$\nabla_{\theta^\mu} J(\theta_t^\mu) \approx \mathbb E_{S_t}[\nabla_{\theta^\mu} S(s,a|\theta^Q)|_{s=S_t,a=\mu(S_t|\theta^\mu)}] \approx \mathbb E_{S_t}[\nabla_a Q(s,a|\theta^Q)|_{s=S_t,a=\mu(S_t)}\nabla_{\theta^\mu}\mu(s|\theta^\mu)|_{s=S_t}]$

$\theta^\mu = \theta^\mu + \alpha\nabla_{\theta^\mu} J$

DDPG算法

- 随机初始化Critic值函数网络$Q(S,A|\theta^Q)$和Actor策略网络$\mu(S|\theta^\mu)$，以及目标网络$\theta^Q$和$\theta^\mu$
- 初始化目标网络Q'和$\mu'$参数：$\theta^{Q'} \leftarrow \theta^{Q},\theta^{\mu'}
 \leftarrow \theta^\mu$;
- 初始化经验回放内存D;
for episode = 1 : M do
    初始化一个随机噪声过程$\mathcal N$;
    初台化状态$s1$;
    for t=1 : T do
        按照当前Actor策略并添加噪声样本$\mathcal N_t$选择一个行动 $A_t = \mu(s_t|\theta^\mu) + \mathcal N_t$
        智能体执行行动A_t，并观测到下一个状态$s_{t+1}$和奖励$R_t$
        将(s_t,A_t,R_t,s_{t+1})存入经验回放内存D;
        从经验回放内存D中随机采样一个mini-batch;
        计算目标值$y_i = R_i + \gamma Q(s_{i+1},\mu'(s_{i+1}|\theta^{\mu'})|\theta^{Q'})$
        更新Critic网络$Q$的参数$\theta^Q$：$min_{\theta^Q} \sum_{i=1}^N (y_i - Q(s_i,a_i|\theta^Q))^2$
        更新Actor网络$\mu$的参数$\theta^\mu$：$min_{\theta^\mu} \sum_{i=1}^N Q(s_i,\mu(s_i|\theta^\mu)|\theta^Q)$
        更新目标网络$Q'$和$\mu'$的参数：$\theta^{Q'} \leftarrow \tau \theta^{Q} + (1-\tau)\theta^{Q'},\theta^{\mu'} \leftarrow \tau \theta^\mu + (1-\tau)\theta^{\mu'}$
        基于经验采样策略梯度法更新Actor网络$\mu$的参数$\theta^\mu$：$min_{\theta^\mu} \sum_{i=1}^N Q(s_i,\mu(s_i|\theta^\mu)|\theta^Q)$
        更新目标网络参数Q'和$\mu'$：
        $\theta^{Q'} \leftarrow \tau \theta^{Q} + (1-\tau)\theta^{Q'},\theta^{\mu'} \leftarrow \tau \theta^\mu + (1-\tau)\theta^{\mu'}$


训练过程：
- 初始化环境
- 按照Actor策略并添加噪声样本来选择行动
- 参与者执行行动产生交互序列，并存储到经验池
- 从经验池中随机采样一个mini-batch
- 计算Critic和Actor的Loss
- 更新Critic和Actor网络
- 软更新两个目标网络

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

