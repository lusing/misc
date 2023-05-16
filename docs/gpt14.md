2023年的深度学习入门指南(14) - 不能只关注模型

在第一篇的时候，我们已经粗略的看了一眼Instruct GPT的步骤图。经历了对于基础知识和编程实战的历练，是时候开始进入细节了。
![](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/instuct.png)

请注意，人类打标的工作是分两部分的：
第一部分是根据问题，由人类写答案。然后用来对大模型进行微调。这一步被称为SFT，有监督的微调。
第二部分是根据模型生成的几个答案，由人类进行排序。人类排序的结果用于训练强化学习的奖励模型。

![](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/%E4%BA%BA%E7%B1%BB%E6%89%93%E6%A0%87%E6%95%B0%E6%8D%AE.png)

第一部分，人类写的答案，分为三种情况：
- 纯文本：就是正常平时大家聊家的一问一答的方式。
- few-shot：这其实就是我们学习prompt工程常用到的方式。给出一个指令，然后给出若干组供模型学习的例子。举个例子，指令是英译中，后面给几对英语和对应的中文的例子。
- 预定义的十大类问题：
    - 生成（generation）
    - 开放式问答（open QA）
    - 封闭式问答（closed QA）
    - 头脑风暴（brainstorming）
    - 聊天（chat）
    - 改写（rewriting）
    - 摘要（summarization）
    - 分类（classification）
    - 提取（extraction）
    - 其他（other）

我们来看几个例子。

1. 头脑风暴：列出五个重拾职业热情的想法
2. 封闭式问答：回答以下问题：地球的形状是什么？ A) 圆形 B) 球形 C) 椭圆形 D) 平面
3. 开放式问答：如何对正弦函数求导？
4. 生成：写一个关于一只棕熊去海滩，与一只海豹交朋友，然后返回家的简短故事

第二部分，由模型生成，而人来进行排序。模型生成4到9个结果，然后由人类来进行排序。
我们将模型生成的结果记为K，则打标者需要进行$\left(\begin{array}{c}K \\ 2\end{array}\right)$次比较。

我们来看下奖励模型的损失函数：
$$
\operatorname{loss}(\theta)=-\frac{1}{\left(\begin{array}{c}
K \\
2
\end{array}\right)} E_{\left(x, y_w, y_l\right) \sim D}\left[\log \left(\sigma\left(r_\theta\left(x, y_w\right)-r_\theta\left(x, y_l\right)\right)\right)\right]
$$

怕公式的同学不要慌，我来分步解释下，这只是形式化地描述了下，其中并不涉及高深的数学或者机器学习的理论。

其中，x表示输入，$y_w$是排序赢了的输出，$y_l$是输的了。
r是奖励函数，参数为$\theta$。

于是，$r_\theta(x, y)$是奖励模型对提示$x$和完成$y$的标量输出。

$\sigma$ 是 sigmoid 函数，将输出值映射到 $(0, 1)$ 之间。这表示一个概率，即在给定一对比较时，优选完成 $y_w$ 被选中的概率。

然后取对数，因为对数损失（或者叫交叉熵损失）是优化分类问题的常用方式，它可以直接优化预测的概率。

$\frac{1}{\left(\begin{array}{c}K \\ 2\end{array}\right)}$ 表示归一化因子，用于在不同大小的比较集合之间进行平衡。

$E_{\left(x, y_w, y_l\right) \sim D}$ 表示在数据集 $D$ 上的期望值。$D$ 是包含人类比较的数据集。如果你不理解数学期望的话，这其实就是平均值。

最后，损失函数是所有$\left(\begin{array}{c}K \\ 2\end{array}\right)$个比较的平均对数损失的相反数，取相反数是因为在优化过程中，我们希望最小化损失，而不是最大化损失。
在训练过程中，模型会学会在给定一对完成时，预测人类标签者更可能选择哪个完成。


总结起来一句人话就是：我们希望选择参数$\theta$，使得$r_\theta(x, y_w) - r_\theta(x, y_l)$尽可能大。也就是让人类选择的结果获取更高的分。

好，前两步搞明白了，我们讲最后的强化学习。这里使用到了PPO算法。
PPO是Proximal Policy Optimization的缩写,是一种策略梯度方法,用于解决强化学习中的策略优化问题。

PPO算法的主要思想是:在策略更新过程中,要确保新旧策略足够接近,以保证学习的稳定性。这是通过引入一个幅度限制来实现的,新策略不能偏离旧策略太多。正如算法名称中“Proximal”所体现的,优化的新策略始终近似于原策略。

PPO算法的主要步骤是:

1. 收集轨迹样本:在环境中采样获得一定数量的轨迹(episode),形成样本集。

2. 计算新旧策略的比率:对样本中的每个时间步,计算新策略π_θ(at|st)与旧策略π_θold(at|st)的比率。

3. 计算剪辑后的比率:将上一步得到的比率剪辑到[1-ε, 1+ε]区间内,避免新策略偏离旧策略太多。

4. 优化策略:通过损失函数 max(Δπ - Δπ剪辑, 0)2 更新策略参数θ,其中Δπ表示新旧策略的比率。

5. 更新旧策略:将当前策略设置为旧策略,为下一轮迭代准备。

6. 重复步骤1-5,直到策略收敛。

PPO算法的主要优点是稳定性高、样本利用率高。相比A2C、ACKTR等方法,PPO可以更快、更稳定地找到较优策略,这使其适合处理许多在线和离线强化学习问题。

PPO已成为实现强化学习的一种流行和有效的方法,在许多环境下取得了state-of-the-art的结果。

回到我们的微调中来，我们构建一个赌博环境，它会呈现一个随机的客户提示，并期望对该提示做出回应。根据提示和回应，它会通过奖励模型生成一个奖励，并结束该回合。此外，我们在每个令牌上添加了来自SFT模型的逐令牌KL惩罚，以减轻对奖励模型的过度优化。价值函数从RM进行初始化。我们将这些模型称为“PPO”。

下面还得先补充一个概念，KL散度。KL散度(KL divergence)是信息论中的一个重要概念, 全名是Kullback-Leibler divergence。它是用来度量两个概率分布之间差异的一种方法。

KL散度的定义如下:对于两个概率分布P和Q,P的KL散度相对于Q是:

$$KL(P||Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

它有以下几个重要性质:

1. KL散度是非对称的,即$KL(P||Q) \neq KL(Q||P)$。它度量的是$P$相对于$Q$的信息量。

2. KL散度总是非负的,$KL(P||Q) \geq 0$。只有当P和Q完全相同时,$KL(P||Q)=0$。

3. KL散度衡量的是两个分布的差异,而不是距离。它表达的是$P$相对于$Q$丢失或获得的信息量。

4. KL散度并不满足三角不等式。

KL散度有许多应用,主要用于量化概率分布之间的差异,或者作为损失函数的一部分。例如:

- 机器翻译中,限制神经机器翻译与统计机器翻译的差异。
- 变分推断中,测量后验分布与概率模型的差异,并最小化。
- 评估生成模型的性能,比如GAN、VAE等。

所以总的来说,KL散度是一个非常有用的工具,用于衡量和控制概率分布之间的差异程度。在许多机器学习模型和算法中有重要应用。

我们还尝试将预训练梯度混合到PPO梯度中，以修复在公共NLP数据集上的性能回归。我们将这些模型称为"PPO-ptx"。在RL训练中，我们最大化以下组合目标函数：

$$
\begin{aligned}
\operatorname{objective}(\phi)= & E_{(x, y) \sim D_{\pi_\phi^{\mathrm{RL}}}}\left[r_\theta(x, y)-\beta \log \left(\pi_\phi^{\mathrm{RL}}(y \mid x) / \pi^{\mathrm{SFT}}(y \mid x)\right)\right]+ \\
& \gamma E_{x \sim D_{\text {pretrain }}}\left[\log \left(\pi_\phi^{\mathrm{RL}}(x)\right)\right]
\end{aligned}
$$
 
这个公式定义了PPO-ptx模型的优化目标(objective function)。它由两部分组成:

1. 强化学习部分:
$$
E_{(x, y) \sim D_{\pi_\phi^{\mathrm{RL}}}}\left[r_\theta(x, y)-\beta \log \left(\pi_\phi^{\mathrm{RL}}(y \mid x) / \pi^{\mathrm{SFT}}(y \mid x)\right)\right] 
$$
这部分的含义是:在强化学习得到的轨迹分布$D_{\pi_\phi^{\mathrm{RL}}}$下,计算每个时间步的奖励$r_\theta(x, y)$与KL惩罚$\beta \log \left(\pi_\phi^{\mathrm{RL}}(y \mid x) / \pi^{\mathrm{SFT}}(y \mid x)\right)$,并取均值。KL惩罚的作用是限制强化学习策略与监督学习策略的差别。
这个项前的系数$\beta$是KL奖励系数，用于控制KL惩罚的强度。

2. 预训练部分:
$$
\gamma E_{x \sim D_{\text {pretrain }}}\left[\log \left(\pi_\phi^{\mathrm{RL}}(x)\right)\right]
$$
这部分的含义是:在预训练数据分布$D_{\text {pretrain }}$下,计算强化学习策略的对数似然,并与超参数$\gamma$相乘,作为预训练损失加入到总的优化目标中。

所以总的来说,这个优化目标在强化学习的同时,也利用预训练数据继续训练模型,以修复强化学习在一些NLP数据集上的性能问题。$\beta$和$\gamma$是控制KL惩罚项和预训练损失项强度的超参数。

