# 2023年的深度学习入门指南(16) - JAX和TPU加速

上一节我们介绍了ChatGPT的核心算法之一的人类指示的强化学习的原理。我知道大家都没看懂，因为需要的知识储备有点多。不过没关系，大模型也不是一天能够训练出来的，也不可能一天就对齐。我们有充足的时间把基础知识先打好。

上一节的强化学习部分没有展开讲的原因是担心大家对于数学知识都快忘光了，而且数学课上学的东西也没有学习编程。这一节我们来引入两个基础工具，一个可以说是各个Python深度学习框架必然绕不过去的NumPy库，另一个是Google开发的可以认为是GPU和TPU版的NumPy库JAX。

学习这两个框架的目的还是补数学课，尤其是数学编程，这次也是TPU首次登场我们的教程部分。当然，也是可以用GPU的。

## 矩阵

NumPy最为核心的功能就是多维矩阵的支持。

我们可以通过`pip install numpy`的方式来安装NumPy，然后在Python中通过`import numpy as np`的方式来引入NumPy库。
但是，NumPy不能支持GPU和TPU加速，对于我们将来要处理的计算来说不太实用，所以我们这里引入JAX库。

![](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/jax_logo_250px.webp)

JAX的安装文档请见[JAX官方文档](https://github.com/google/jax#installation)

之前我们多次使用CUDA来进行GPU加速了，这里我们不妨来看看TPU的加速效果。
TPU只有Google一家有，我们只能买到TPU的云服务，不过，我们可以使用Google Colab来使用TPU。

在Colab上，已经安装好了JAX和TPU的运行时。我们运行下面的代码即可激活TPU:

```python
import jax.tools.colab_tpu
jax.tools.colab_tpu.setup_tpu()
```

我们来看看有多少个TPU设备可以使用:

```python
print(jax.device_count())
print(jax.local_device_count())
print(jax.devices())
```

输出结果如下：
```
8
8
[TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), TpuDevice(id=1, process_index=0, coords=(0,0,0), core_on_chip=1), TpuDevice(id=2, process_index=0, coords=(1,0,0), core_on_chip=0), TpuDevice(id=3, process_index=0, coords=(1,0,0), core_on_chip=1), TpuDevice(id=4, process_index=0, coords=(0,1,0), core_on_chip=0), TpuDevice(id=5, process_index=0, coords=(0,1,0), core_on_chip=1), TpuDevice(id=6, process_index=0, coords=(1,1,0), core_on_chip=0), TpuDevice(id=7, process_index=0, coords=(1,1,0), core_on_chip=1)]
```

说明我们有8个TPU设备可以用。

![](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/Tensor_Processing_Unit_3.0.jpg)

下面我们就用jax.numpy来代替numpy来使用。

NumPy最重要的功能就是多维矩阵的支持。我们可以通过`np.array`来创建一个多维矩阵。

我们先从一维的向量开始：

```python
import jax.numpy as jnp
a1 = jnp.array([1,2,3])
print(a1)
```

然后我们可以使用二维数组来创建一个矩阵：

```python
a2 = jnp.array([[1,2],[0,4]])
print(a2)
```

矩阵可以统一赋初值。zeros函数可以创建一个全0的矩阵，ones函数可以创建一个全1的矩阵，full函数可以创建一个全是某个值的矩阵。

比如给10行10列的矩阵全赋0值，我们可以这样写：
```python
a3 = jnp.zeros((10,10))
print(a3)
```

全1的矩阵：
```python
a4 = jnp.ones((10,10))
```

全赋100的：
```python
a5 = jnp.full((10,10),100)
```

另外，我们还可以通过linspace函数生成一个序列。linpsace函数的第一个参数是序列的起始值，第二个参数是序列的结束值，第三个参数是序列的长度。比如我们可以生成一个从1到100的序列，长度为100：

```python
a7 = jnp.linspace(1,100,100) # 从1到100，生成100个数
a7.reshape(10,10)
print(a7)
```

最后，JAX给矩阵生成随机值的方式跟NumPy并不一样，并没有jnp.random这样的包。我们可以使用jax.random来生成随机值。JAX的随机数生成函数都需要一个显式的随机状态作为第一个参数，这个状态由两个无符号32位整数组成，称为一个key。用一个key不会修改它，所以重复使用同一个key会得到相同的结果。如果需要新的随机数，可以使用jax.random.split()来生成新的子key。

```python
from jax import random
key = random.PRNGKey(0) # a random key
key, subkey = random.split(key) # split a key into two subkeys
a8 = random.uniform(subkey,shape=(10,10)) # a random number using subkey
print(a8)
```

## 范数

范数（Norm）是一个数学概念，用于测量向量空间中向量的“大小”。范数需要满足以下性质：

- 非负性：所有向量的范数都大于或等于零，只有零向量的范数为零。
- 齐次性：对任意实数λ和任意向量v，有||λv|| = |λ| ||v||。
- 三角不等式：对任意向量u和v，有||u + v|| ≤ ||u|| + ||v||。
在实际应用中，范数通常用于衡量向量或矩阵的大小，比如在机器学习中，范数常用于正则化项的计算。

常见的范数有：

- L0范数：向量中非零元素的个数。
- L1范数：向量中各个元素绝对值之和，也被称为曼哈顿距离。
- L2范数：向量中各个元素的平方和然后开方，也被称为欧几里得距离。
- 无穷范数：向量中各个元素绝对值的最大值。
需要注意的是，L0范数并不是严格意义上的范数，因为它违反了齐次性。但是在机器学习中，L0范数常用于衡量向量中非零元素的个数，因此也被称为“伪范数”。

我们先从计算一个一维向量的L1范数开始，不要L1范数这个名字给吓到，其实就是绝对值之和：
```python
norm10_1 = jnp.linalg.norm(a10,ord=1)
print(norm10_1)
```

结果不出所料就是6.

下面我们再看L2范数，也就是欧几里得距离，也就是平方和开方：

```python
a10 = jnp.array([1, 2, 3])
norm10 = jnp.linalg.norm(a10)
print(norm10)
```

根据L2范数的定义，我们可以手动计算一下：norm10 = jnp.sort(1 + 2*2 + 3*3) = 3.7416573.

我们可以看到，上面的norm10的值跟我们手动计算的是一样的。

下面我们来计算无穷范数，其实就是最大值：
```python
norm10_inf = jnp.linalg.norm(a10, ord = jnp.inf)
print(norm10_inf)
```

结果为3.

我们来算一个大点的巩固一下：

```python
a10 = jnp.linspace(1,100,100) # 从1到100，生成100个数
n10 = jnp.linalg.norm(a10,ord=2)
print(n10)
```

这个结果为581.67865.

## 逆矩阵

对角线是1，其它全是0的方阵，我们称为单位矩阵。在NumPy和JAX中，我们用eye函数来生成单位矩阵。

既然是方阵，就不用跟行和列两个值了，只需要一个值就可以了，这个值就是矩阵的行数和列数。用这一个值赋给eye函数的第一个参数，就可以生成一个单位矩阵。

下面我们来复习一下矩阵乘法是如何计算的。

对于矩阵A的每一行，我们需要与矩阵B的每一列相乘。这里的“相乘”意味着取A的一行和B的一列，然后将它们的对应元素相乘，然后将这些乘积相加。这个和就是结果矩阵中相应位置的元素。

举个例子，假设我们有两个2x2的矩阵A和B：

```
A = 1 2     B = 4 5
    3 4         6 7
```

我们可以这样计算矩阵A和矩阵B的乘积：

```
(1*4 + 2*6) (1*5 + 2*7)     16 19
(3*4 + 4*6) (3*5 + 4*7) =  34 43
```

我们用JAX来计算一下：
```python
ma1 = jnp.array([[1,2],[3,4]])
ma2 = jnp.array([[4,5],[6,7]])
ma3 = jnp.dot(ma1,ma2)
print(ma1)
print(ma2)
print(ma3)
```

输出为：
```
[[1 2]
 [3 4]]
[[4 5]
 [6 7]]
[[16 19]
 [36 43]]
```

如果A*B=I，I为单位矩阵，那么我们称B是A的逆矩阵。

我们可以用inv函数来计算矩阵的逆矩阵。

```python
ma1 = jnp.array([[1,2],[3,4]])
inv1 = jnp.linalg.inv(ma1)
print(inv1)
```

输出的结果为：
```
[[-2.0000002   1.0000001 ]
 [ 1.5000001  -0.50000006]]
 ```

## 导数与梯度

导数是一个函数在某点处的变化率，用于描述函数在该点处的变化率。导数可以表示函数在该点处的斜率，即函数在该点处的陡峭程度。

梯度(gradient)是一个向量，表示函数在该点处的方向导数沿着该方向取得最大值。梯度可以表示函数在该点处的变化最快和变化率最大的方向。在单变量的实值函数中，梯度可以简单理解为导数。

JAX作为支持深度学习的框架，对于梯度的支持是被优先考虑的。我们可以使用jax.grad函数来计算梯度。针对一个一元函数，梯度就是导数。我们可以用下面的代码来计算sin函数在x=1.0处的梯度：

```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.sin(x)

# 计算 f 在 x=1.0 处的梯度
grad_f = jax.grad(f)
print(grad_f(1.0))
```

我们如果每次都沿着梯度方向前进，那么我们就可以找到函数的极值。这种采用梯度方向前进的方法，就是梯度下降法。梯度下降法是一种常用的优化算法，它的核心思想是：如果一个函数在某点的梯度值为正，那么函数在该点沿着梯度方向下降的速度最快；如果一个函数在某点的梯度值为负，那么函数在该点沿着梯度方向上升的速度最快。因此，我们可以通过不断地沿着梯度方向前进，来找到函数的极值。

那么，梯度下降法有什么作用呢？我们可以用梯度下降法来求解函数的最小值。我们可以用下面的代码来求解函数$f(x)=x^2$的最小值：

```python
import jax
import jax.numpy as jnp

def f(x):
    return x ** 2

grad_f = jax.grad(f)

x = 2.0  # 初始点
learning_rate = 0.1  # 学习率
num_steps = 100  # 迭代步数

for i in range(num_steps):
    grad = grad_f(x)  # 计算梯度
    x = x - learning_rate * grad  # 按负梯度方向更新 x

print(x)  # 打印最终的 x 值，应接近 0（函数的最小值）
```

我这一次运行的结果是4.0740736e-10. 也就是说，我们用梯度下降法求解函数$f(x)=x^2$的最小值，最终得到的x值接近于0，也就是函数的最小值。

其中，学习率（或称为步长）是一个正数，用于控制每一步更新的幅度。学习率需要仔细选择，过大可能导致算法不收敛，过小可能导致收敛速度过慢。

## 概率

唤醒完线性代数和高等数学的一些记忆之后，最后我们来回顾一下概率论。

我们还是从扔硬币说起。我们知道，假设一枚硬币是均匀的，那么只要扔的次数足够多，正面朝上的次数就会接近于总次数的一半。

这种只有两种可能结果的随机试验，我们给它起个高大上的名字叫做伯努利试验(Bernoulli trial)。

下面我们就用JAX的伯努利分布来模拟一下扔硬币的过程。

```python
import jax
import time
from jax import random

# 生成一个形状为 (10, ) 的随机矩阵，元素取值为 0 或 1，概率为 0.5
key = random.PRNGKey(int(time.time()))
rand_matrix = random.bernoulli(key, p=0.5, shape=(10, ))
print(rand_matrix)
mean_x = jnp.mean(rand_matrix)
print(mean_x)
```

mean函数用来求平均值，也叫做数学期望。

打印的结果可能是0.5，也可能是0.3，0.8等等。这是因为我们只扔了10次硬币，扔的次数太少了，所以正面朝上的次数不一定接近于总次数的一半。

这是其中一次0.6的结果：
```
[ True  True  True  True False False  True False False  True]
0.6
```

多跑几次，出现0.1，0.9都不稀奇：
```
[False False False False False False False False False  True]
0.1
```

当我们把shape改成100，1000，10000等更大的数之后，这个结果就离0.5越来越近了。

下面再复习下表示偏差的两个值：
- 方差（Variance）：方差是度量数据点偏离平均值的程度的一种方式。换句话说，它描述了数据点与平均值之间的平均距离的平方。
- 标准差（Standard Deviation）：标准差是方差的平方根。因为方差是在平均偏差的基础上平方得到的，所以它的量纲（单位）与原数据不同。为了解决这个问题，我们引入了标准差的概念。标准差与原数据有相同的量纲，更便于解释。

这两个统计量都反映了数据分布的离散程度。方差和标准差越大，数据点就越分散；反之，方差和标准差越小，数据点就越集中。

我们可以用JAX的var函数来计算方差，用std函数来计算标准差。

```python
import jax
import time
from jax import random

# 生成一个形状为 (1000, ) 的随机矩阵，元素取值为 0 或 1，概率为 0.5
key = random.PRNGKey(int(time.time()))
rand_matrix = random.bernoulli(key, p=0.5, shape=(1000, ))
#print(rand_matrix)
mean_x = jnp.mean(rand_matrix)
print(mean_x)
var_x = jnp.var(rand_matrix)
print(var_x)
std_x = jnp.std(rand_matrix)
print(std_x)
```

最后我们来复习一下之前讲到的信息量。我们来思考一个问题，如何能让伯努利分布的平均信息量最大？

我们先构造两个特殊情况，比如如果p=0，那么我们就永远不会得到正面朝上的结果，这个时候我们就知道了结果，信息量为0。如果p=1，那么我们就永远不会得到反面朝上的结果，这个时候我们也知道了结果，信息量也为0。

如果p=0.01，能给我们带来的平均信息量仍然不大，因为基本上我们可以盲猜结果是背面朝上的，偶然出现的正面朝上的结果，虽然带来了较大的单次信息量，但是出现的概率太低了，所以平均信息量仍然不大。

而如果p=0.5，我们就完全猜不到结果是正面朝上还是背面朝上，这个时候我们得到的平均信息量最大。

当然，这只是定性的分析，我们还需要给出一个定量的公式：

$$
H(X) = - \sum_{x \in X} p(x) \log_2 p(x)
$$

![](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/440px-Entropy_flip_2_coins.jpg)

```python
import jax.numpy as jnp

# 计算离散型随机变量 X 的平均信息量
def avg_information(p):
    p = jnp.maximum(p, 1e-10)
    return jnp.negative(jnp.sum(jnp.multiply(p, jnp.log2(p))))

# 计算随机变量 X 取值为 0 和 1 的概率分别为 0.3 和 0.7 时的平均信息量
p = jnp.array([0.3, 0.7])
avg_info = avg_information(p)
print(avg_info)
```

我们试几次计算可以得到，当p为0.3时，平均信息量是0.8812325;当p为0.01时，平均信息量为0.08079329；当p为0.5时，平均信息量为1.0，达到最大。

如果嫌使用Python函数的计算慢，我们可以调用JAX的jit函数来加速。我们只需要在函数定义的前面加上@jit即可。

```python
import jax.numpy as jnp
from jax import jit

# 计算离散型随机变量 X 的平均信息量
@jit
def avg_information(p):
    p = jnp.maximum(p, 1e-10)
    return jnp.negative(jnp.sum(jnp.multiply(p, jnp.log2(p))))

# 计算随机变量 X 取值为 0 和 1 的概率分别为 0.3 和 0.7 时的平均信息量
p = jnp.array([0.01, 0.99])
avg_info = avg_information(p)
print(avg_info)
```

## 小结

上面我们选取了一些线性代数，高等数学和概率论的知识点，来唤醒大家的记忆。同时，我们也介绍了它们在JAX上的实现和加速。
虽然我们的例子都不起眼，但它们是确确实实在TPU上跑起来的。

大模型虽然提供了很强的能力，但是我们仍然要花充足的时间在基本功上。硬件和框架都在日新月部分，但是数学基础知识的进化是非常缓慢的，投入产出比很高。有了扎实的基本功之后，框架和新硬件就是边用边学就可以了。
