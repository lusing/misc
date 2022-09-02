# Proximal Policy Optimization Algorithms

## Abstract

We propose a new family of policy gradient methods for reinforcement learning, which alternate between sampling data through interaction with the environment, and optimizing a “surrogate” objective function using stochastic gradient ascent. Whereas standard policy gradient methods perform one gradient update per data sample, we propose a novel objective function that enables multiple epochs of minibatch updates. The new methods, which we call proximal policy optimization (PPO), have some of the benefits of trust region policy optimization (TRPO), but they are much simpler to implement, more general, and have better sample complexity (empirically). Our experiments test PPO on a collection of benchmark tasks, including simulated robotic locomotion and Atari game playing, and we show that PPO outperforms other online policy gradient methods, and overall strikes a favorable balance between sample complexity, simplicity, and wall-time.

我们提出了一种用于强化学习的新策略梯度方法家族，通过与环境的交互在采样数据之间交替，并使用随机梯度上升优化“替代”目标函数。标准策略梯度方法对每个数据样本进行一次梯度更新，而我们提出了一个新的目标函数，使多个时期的小批量更新成为可能。这种新方法被称为近端策略优化(PPO)，具有信任区域策略优化(TRPO)的一些优点，但其实现更简单，更通用，并具有更好的样本复杂度(实证)。我们的实验在一系列基准任务上测试了PPO，包括模拟机器人运动和雅达利游戏玩法，我们表明，PPO优于其他在线策略梯度方法，总体上在样本复杂性、简单性和时间之间取得了良好的平衡。

## 1 Introduction

In recent years, several different approaches have been proposed for reinforcement learning with neural network function approximators. The leading contenders are deep Q-learning [Mni+15], “vanilla” policy gradient methods [Mni+16], and trust region / natural policy gradient methods [Sch+15b]. However, there is room for improvement in developing a method that is scalable (to large models and parallel implementations), data efficient, and robust (i.e., successful on a variety of problems without hyperparameter tuning). Q-learning (with function approximation) fails on many simple problems1 and is poorly understood, vanilla policy gradient methods have poor data effiency and robustness; and trust region policy optimization (TRPO) is relatively complicated, and is not compatible with architectures that include noise (such as dropout) or parameter sharing (between the policy and value function, or with auxiliary tasks).

近年来，人们提出了几种不同的利用神经网络函数逼近器进行强化学习的方法。领先的竞争者是深度Q-learning [Mni+15]，“香草”策略梯度方法[Mni+16]，和信任区域/自然策略梯度方法[Sch+15b]。然而，在开发一种可伸缩(对大型模型和并行实现)、数据高效和健壮(即，在不进行超参数调优的情况下成功解决各种问题)的方法方面还有改进的空间。q -学习(带有函数逼近)在许多简单的问题上失败1和理解不足，传统的策略梯度方法数据效率和鲁棒性较差;信任域策略优化(TRPO)相对复杂，不兼容包含噪声(如dropout)或参数共享(策略和值函数之间，或与辅助任务)的架构。

This paper seeks to improve the current state of affairs by introducing an algorithm that attains the data efficiency and reliable performance of TRPO, while using only first-order optimization. We propose a novel objective with clipped probability ratios, which forms a pessimistic estimate (i.e., lower bound) of the performance of the policy. To optimize policies, we alternate between sampling data from the policy and performing several epochs of optimization on the sampled data.

本文试图通过引入一种算法来改善当前状态，该算法在仅使用一阶优化的情况下，就能达到TRPO的数据效率和可靠性能。我们提出了一个新的目标与剪切概率比，形成悲观估计(即下限)的政策的表现。为了优化策略，我们轮流从策略中采样数据，并对采样数据执行多个时段的优化。

Our experiments compare the performance of various different versions of the surrogate objective, and find that the version with the clipped probability ratios performs best. We also compare PPO to several previous algorithms from the literature. On continuous control tasks, it performs better than the algorithms we compare against. On Atari, it performs significantly better (in terms of sample complexity) than A2C and similarly to ACER though it is much simpler.

我们的实验比较了不同版本的替代目标的性能，发现具有剪切概率比的版本性能最好。我们还将PPO算法与文献中先前的几个算法进行了比较。在连续控制任务中，它的性能优于我们所比较的算法。在Atari上，它的表现(就样本复杂度而言)比A2C要好得多，与ACER类似，尽管它要简单得多。

## 2 Background: Policy Optimization

## 3 Clipped Surrogate Objective

## 4 Adaptive KL Penalty Coefficient

## 5 Algorithm

## 6 Experiments

## 7 Conclusion

We have introduced proximal policy optimization, a family of policy optimization methods that use multiple epochs of stochastic gradient ascent to perform each policy update. These methods have the stability and reliability of trust-region methods but are much simpler to implement, requiring only few lines of code change to a vanilla policy gradient implementation, applicable in more general settings (for example, when using a joint architecture for the policy and value function), and have better overall performance.

我们引入了近端政策优化，这是一组利用多个随机梯度上升时期来执行每次政策更新的政策优化方法。这些方法具有信任区域方法的稳定性和可靠性，但实现起来要简单得多，只需要对普通策略梯度实现进行几行代码更改，适用于更一般的设置(例如，当为策略和值函数使用联合架构时)，并且具有更好的总体性能。
