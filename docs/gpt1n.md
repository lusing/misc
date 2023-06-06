# 2023年的深度学习入门指南(15) - JAX和TPU加速

上一节我们介绍了ChatGPT的核心算法之一的人类指示的强化学习的原理。我知道大家都没看懂，因为需要的知识储备有点多。不过没关系，大模型也不是一天能够训练出来的，也不可能一天就对齐。我们有充足的时间把基础知识先打好。

上一节的强化学习部分没有展开讲的原因是担心大家对于数学知识都快忘光了，而且数学课上学的东西也没有学习编程。这一节我们来引入两个基础工具，一个可以说是各个Python深度学习框架必然绕不过去的NumPy库，另一个是Google开发的可以认为是GPU和TPU版的NumPy库JAX。

学习这两个框架的目的还是补数学课，尤其是数学编程，这次也是TPU首次登场我们的教程部分。当然，也是可以用GPU的。

## 矩阵

NumPy最为核心的功能就是多维矩阵的支持。

我们可以通过`pip install numpy`的方式来安装NumPy，然后在Python中通过`import numpy as np`的方式来引入NumPy库。
但是，NumPy不能支持GPU和TPU加速，对于我们将来要处理的计算来说不太实用，所以我们这里引入JAX库。
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

```