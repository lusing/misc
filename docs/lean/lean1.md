# 面向程序员的Lean4教程(1) - 像传统编程语言一样使用Lean

随着大模型技术应用的深入，AI与数学的结合也变得越来越流行。

菲尔兹奖得主、加州大学洛杉矶分校教授陶哲轩对于Lean和人工智能的看好，也让Lean4逐渐有破圈的趋势。

不过，Lean4的小圈子里流传的，不仅仅是普通程序员认为的数学，比如高等数学、线性代数、概率论、离散数学等等，上来就被柯里-霍华德同构等概念砸晕，然后觉得Lean4哪怕跟Isabelle、Coq等定理证明器相比，也是相当不同。

另外，Lean4还有一个著名的数学库Mathlib4，这是一个目前150万行量级的数学库，包含了大量的数学定理和证明，这是一笔宝贵的财富。尤其是在大模型时代，大量的数学证明语料是非常珍贵的。

其实，Lean 4的文档中一再强调，不仅是一个定理证明器，还是一个通用的编程语言。它支持函数式编程、依赖类型和元编程，用户可以用 Lean 4编写实际的程序。

下面我们就暂时忘掉定理证明的部分，从程序员的角度来看看Lean4。

## Lean4的安装

Lean4同Rust类似，有一个类似于Cargo的包管理工具，叫做elan。elan是一个用于管理Lean 4的工具链的工具，它可以安装Lean 4的最新版本，也可以安装指定版本。

然后，Lean4使用lake作为构建工具，它可以自动下载依赖项，编译源代码，运行测试等。

下面我们隆重介绍适合国内使用的glean工具，文档在：https://mirror.sjtu.edu.cn/docs/elan。

我们通过下面的链接下载glean: http://mirror.sjtu.edu.cn/elan/?mirror_intel_list.

比如当前的版本：v0.1.18，下载地址是 https://s3.jcloud.sjtu.edu.cn/899a892efef34b1b944a19981040f55b-oss01/elan/glean/releases/download/v0.1.18/mirror_clone_list.html

根据不同的操作系统，可以下载不同的版本，比如我用Windows系统，就下载 https://s3.jcloud.sjtu.edu.cn/899a892efef34b1b944a19981040f55b-oss01/elan/glean/releases/download/v0.1.18/glean_Windows_x86_64.zip。

glean有三个主要功能：
- 下载elan
- 下载lean4 toolchain
- 下载库的依赖，主要是Mathlib4


可以通过glean -install elan命令安装elan，版本号还是到 http://mirror.sjtu.edu.cn/elan/?mirror_intel_list 中查看。

```
glean -install elan -version v4.0.0-rc1
```

然后，可以通过glean -install lean命令安装lean4 toolchain，版本号还是到 http://mirror.sjtu.edu.cn/elan/?mirror_intel_list 中查看。

```
glean -install lean -version v4.15.0
```

有了elan之后，我们就可以通过lake new命令创建一个新的项目。

```
lake new math666
```

进入math666目录后，我们可以再次调用glean来下载依赖，目前我们没有什么依赖，所以这一步没有什么作用。

然后我们再调用lake update来确认下依赖更新成功：

```
lake update
```

接着我们调用lake build来构建项目：

```
lake build
```

最后我们就可以调用lake exec来运行项目：

```
lake exec math666
```

输出如下：

```
Hello, world!
```

我们来看一下math666的主函数：

```ml
def main : IO Unit :=
  IO.println s!"Hello, {hello}!"
```

这里的IO Unit是一个Monad，它表示一个不接受输入，只输出Unit的函数。而IO.println是一个函数，它接受一个字符串，然后输出到控制台。

对于没有函数式编程经验的程序员来说，这里的Monad可能有点陌生，不过不用担心，只要理解函数式语言要求没有副作用，输入输出等有副作用的操作都要放在Monad里，就可以了。

我们最后看一下lake.toml文件，这是一个类似于Makefile的文件，用来描述项目的结构。

```toml
name = "math666"
version = "0.1.0"
defaultTargets = ["math666"]

[[lean_lib]]
name = "Math666"

[[lean_exe]]
name = "math666"
root = "Main"
```

主文件是Main.lean，我们可以看到它的内容是：

```ocaml
import Math666

def main : IO Unit :=
  IO.println s!"Hello, {hello}!"
```

编译后生成的可执行文件是math666。

我们可以照葫芦画瓢，再加一条输出语句：

```ocaml
import Math666

def main : IO Unit := do
  IO.println s!"Hello, {hello}!"
  IO.println "My First Lean Program"
```

先lake build，再lake exec也好，还是直接用lake exec也好，都可以看到我们新增的输出语句起作用了。

## Mathlib4

下面我们要请出我们的另一个主角Mathlib4。这一步开始有点复杂了，但是对于身经百战的程序员来说，并不是太大的问题。

增加对Mathlib4的依赖非常简单，我们需要修改lake.toml文件，增加Mathlib4的依赖就可以了：

```toml
name = "math666"
version = "0.1.0"
defaultTargets = ["math666"]

[[lean_lib]]
name = "Math666"

[[lean_exe]]
name = "math666"
root = "Main"

[[require]]
name = "mathlib"
scope = "leanprover-community"
rev = "v4.15.0"
```

国内用户可以先调用glean来下载依赖库。也可以哪里出错了，用上海交大的源来代替。

比如mathlib4下载失败，可以换用 https://mirror.sjtu.edu.cn/git/lean4-packages/mathlib4。

增加了Mathlib4的依赖后，我们再次调用lake update来更新依赖：

```
lake update
```

主要安装一些自动化工具，图形化工具，元编程工具等。

下面，为了缩短编译几千个文件的时间，我们可以下载预编译的二进制文件，这样就不用每次都重新编译了。

```
lake exe cache get
```

在我目前所用的4.15.0版本中，要下载 5826 个缓存的预编译文件。

最后，我们运行lake build：

```
lake build
```

因为下载了缓存，所以编译速度很快。

下面我们使用Mathlib4来进行有理数的运算。

```ml
import Math666
import Mathlib.Data.Rat.Defs

def a : ℚ := 1 / 2
def b : ℚ := 3 / 4
def c : ℚ := a + b

def main : IO Unit := do
  IO.println s!"Hello, {hello}!"
  IO.println "My First Lean Program"
  IO.println a
  IO.println b
  IO.println c
```

再次lake build，然后lake exec math666, 输出结果如下：

```
Hello, world!
My First Lean Program
1/2
3/4
5/4
```

## Lean 4的基本数值类型

下面我们就从Lean4的基本数据类型开始说起。

Lean4作为一门有数学味儿的语言，支持自然数的数据类型。自然数是大于等于0的正整数，可以通过succ函数计算自然数的后继。

如大家所熟悉的，Lean4也是使用def来绑定全局值，使用let来绑定局部值。

```ml
  let n1 : Nat := 1
  let n2 := Nat.succ n1
  IO.println n2
```

增加负整数，就构成了Int类型。

```ml
  let i1 : Int := 1
  let i2 : Int := -i1
  IO.println i2
```

与别的语言的不限长度的整数一样，Lean4的Int类型也是不限长度的。

```ml
  let i3 : Int := 123456789000000
  let i4 : Int := 4294967296
  let i5 := i3 + i4
  IO.println i5
```

进一步如果要支持有理数的话，那就需要Mathlib4的支持了。像我们前面的例子一样：

```ml
  let a : ℚ := 1 / 2
  let b : ℚ := 3 / 4
  let c : ℚ := a + b
  IO.println c
```

ℚ是Rat，可以通过\Q或者\Rat来输入。

Lean4的浮点数就没有整数那么神奇了，它是符合IEEE 754 双精度浮点标准浮点数。

例：

```ocaml
  let f1 : Float := 2.0
  let f2 := Float.sqrt f1
  IO.println f2
```

比如搞一个超出范围的浮点数：

```ocaml
  let f3 : Float := 1.03e+400
  IO.println f3
```

打印出来就是：

```
inf
```

我们可以用一个布尔值来表示真假，比如判断上面的浮点数是否是无穷大：

```ocaml
  let b1 := Float.isInf f3
  IO.println b1
```

## Lean4的基本流程控制

Lean4的绑定都是常量，不可变的。对于变量，在Lean4中是当作副作用来处理的。具体实现，我们可以用Ref来实现，通过IO.mkRef来创建一个Ref，然后通过set和get来设置和获取值。

```ml
  let mutVar ← IO.mkRef 0
  mutVar.set 1
  IO.println (← mutVar.get)
```

请注意，“←”是单子提取操作符，它可以从一个Monad中提取值。mutVar.get返回一个IO Nat，通过“←”可以提取出Nat。

当然，我们也可以指定Ref的类型：

```ml
  let mutIVar ← IO.mkRef (1: Int)
  mutIVar.set (100: Int)
  IO.println (← mutIVar.get)
```

也可以采用let mut来定义可变变量，例：

```ml
  let mut i := 5
  i := i + 1
```

Lean 4提供了经典的if-then-else语句，例：

```ml
def checkPass(n: Nat) : Bool :=
  if n >= 60 then
    true
  else
    false
```

Lean 4也提供了经典的while循环，例：

```ml
  let mut i := 100
  while i > 90 do
    IO.println i
    i := i - 1
```

针对集合，Lean 4提供了for循环，例：

```ml
  let numbers := [1, 2, 3, 4, 5]
  for n in numbers do
    IO.println n
```

同样，Lean 4也提供了break和continue语句，break用于提前退出，continue用于继续下一次循环，例：

```ml
def searchNumber (target : Nat) : IO Unit := do
  let numbers := [1, 2, 3, 4, 5]
  for n in numbers do
    if n = target then
      IO.println "Found it!"
      break
    else
      continue
```

在函数中，我们可以也使用return来提前返回。

总体来说，结构化编程的基本流程控制在Lean 4中都有支持。

## 定义函数

如前面所看到的，Lean 4也使用def来定义函数。函数的类型可以通过冒号来指定，也可以通过类型推导来推断。例：

```ml
def inc (i : Int) : Int := i + 1
```

也可以不指定类型，让Lean 4自己推断，例：

```ml
def incf (f : Float) := f + 1.0
```

当有多个参数时，可以使用括号来分组，例：

```ml
def maximum (n : Nat) (k : Nat) : Nat :=
  if n < k then
    k
  else n
```

## 定义结构体

Lean 4也支持结构体，通过structure关键字来定义。例：

```ml
structure Point :=
  (x : Int)
  (y : Int)
```

然后我们可以通过结构体赋值来创建一个Point对象，例：

```ml
let p1 : Point := { x := 0.0, y := 0.0 }
```

也可以通过Point.mk来创建一个Point对象，例：

```ml
  let p2 := Point.mk 1.0 1.0
```

我们可以通过点号来访问结构体的成员，例：

```ml
  IO.println p1.x
  IO.println p2.y
```

如果我们想将结构体的内容打印出来，可以实现ToString类型类来实现。类型类有点像其他语言的接口或者是Go语言的Trait。
使用 instance 关键字可以为特定类型实现类型类的实例。实例中需要提供类型类中定义方法的具体实现。

比如我们要将ToString类型类实现为Point类型，并实现其定义的toString方法：

```ml
instance : ToString Point where
  toString p := s!"Point(x = {p.x}, y = {p.y})"
```

然后我们就可以通过IO.println来打印Point对象了，例：

```ml
  IO.println p1
  IO.println p2
```

同样，我们可以生成一个OfNat类型类的实例，这样我们就可以用一个自然数来初始化Point对象了。

```ml
instance : OfNat Point n where
  ofNat := { x := n.toFloat, y := n.toFloat }
```

然后我们就可以用一个自然数来初始化Point对象了，例：

```ml
  let p3 : Point := 3
  IO.println p3
```

输出如下：

```
Point(x = 3.000000, y = 3.000000)
```

## 小结

通过本节的学习，我们可以看到，完全可以像结构化编程语言一样使用Lean 4。Lean 4支持if-then-else、while、for等基本流程控制，支持def来定义函数，支持structure来定义结构体，支持类型类等等常规编程语言的特性。

同时，我们还学习了引入巨大的Mathlib4库，来支持更面向数学的功能。

通篇不用说定理证明了，连函数式编程的内容都基本没有提到，尽管用到了Monad，但是也是为了支持副作用。

到此为止，大家应该对于Lean 4祛魅了一些，不再觉得它是一个高不可攀的定理证明器，而是一个可以用来写实际程序的编程语言。唯一的困难是不熟悉，这通过大家实际编程应该很快可以解决。

下一节我们仍然不谈深入的话题，而是进一步讲解Lean 4的普通编程语言特性。
