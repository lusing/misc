面向程序员的Lean 4教程(4) - 结构体

结构体也是程序员们的老朋友了，它的作用是定义一个数据类型，并给这个数据类型定义一些字段。

## 结构体

在Lean中，结构体使用`structure`关键字定义。

```lean
structure Point where
    x : Nat
    y : Nat
```

Lean 4 会为结构体自动生成默认构造函数，名称是 `结构体名.mk`。也可以通过定义函数来自定义构造函数。

使用 { 字段名 := 值, ... } 的语法可以更方便地创建结构体实例。

我们来看一个例子：

```lean
-- 定义五行枚举类型
inductive FiveElement : Type where
  | Wood   -- 木
  | Fire   -- 火
  | Earth  -- 土
  | Metal  -- 金
  | Water  -- 水
  deriving Repr

-- 为 FiveElement 定义 ToString 实例
instance : ToString FiveElement where
  toString
  | FiveElement.Wood  => "木"
  | FiveElement.Fire  => "火"
  | FiveElement.Earth => "土"
  | FiveElement.Metal => "金"
  | FiveElement.Water => "水"

-- 定义阴阳枚举类型
inductive YinYang : Type where
  | Yin  -- 阴
  | Yang -- 阳
  deriving Repr

-- 为 YinYang 定义 ToString 实例
instance : ToString YinYang where
  toString
  | YinYang.Yin  => "阴"
  | YinYang.Yang => "阳"

-- 定义天干结构体
structure HeavenlyStem where
  name : String       -- 天干名称
  element : FiveElement -- 五行属性
  yinYang : YinYang   -- 阴阳属性
  deriving Repr

```

我们来看两种创建结构体实例的方法：

```lean
def jia : HeavenlyStem := { name := "甲", element := FiveElement.Wood, yinYang := YinYang.Yang }
def yi : HeavenlyStem := HeavenlyStem.mk "乙" FiveElement.Wood YinYang.Yin
```

## 结构体的继承

结构体可以继承另一个结构体，从而获得父结构体的字段和方法。

比如，我们可以为2D的点添加z坐标，从而获得3D的点。

```lean
structure Point where
  x : Float
  y : Float
  deriving Repr

structure Point3D extends Point where
  z : Float
  deriving Repr

def redPoint : Point3D :=
  { x := 1.0, y := 2.0, z := 3.0 }

#eval redPoint.x      -- 输出: 1.0
#eval redPoint.y      -- 输出: 2.0
#eval redPoint.z      -- 输出: 3.0

def point2d : Point := { x := 0.0, y := 0.0 }
def point3d : Point3D := { point2d with z := 4.0 }

#eval point3d  -- 输出: { x := 0.0, y := 0.0, z := 4.0 }
```

函数是Lean4中的一等公民，自然也可以是结构体的一部分。

```lean
structure Point where
  x : Float
  y : Float
  distanceToOrigin : Float := Float.sqrt (x * x + y * y)
  deriving Repr
```
  
定义在结构体的外面也不影响使用，比如我们给点增加一个计算曼哈顿距离的函数。

```lean
def Point.manhattanDistanceToOrigin (p : Point) : Float :=
  Float.abs p.x + Float.abs p.y
```

调用方法是一样的：

```lean
#eval point2d.distanceToOrigin  -- 输出: 5.0
#eval point2d.manhattanDistanceToOrigin  -- 输出: 7.0

```

## 伪装成基本类型的结构体

上一节我们学习了一些伪装成基本类型的归纳类型。还有一些基本类型没有介绍到，其实它们都是伪装成基本类型的结构体。

我们先看看Float类型：

```
info: .\.\.\.\Test3\Basic.lean:47:0: structure Float : Type
number of parameters: 0
fields:
  Float.val : floatSpec.float
constructor:
  Float.mk (val : floatSpec.float) : Float
```

Float类型是一个结构体，它有一个字段val，类型是floatSpec.float。

FloatSpec又是个啥呢，还是个结构体：

```lean
structure FloatSpec where
  float : Type
  val   : float
  lt    : float → float → Prop
  le    : float → float → Prop
  decLt : DecidableRel lt
  decLe : DecidableRel le
```

val是float类型，而float是Type类型。这是根据系统不同定义的类型，总之是符合IEEE 754标准的浮点数。

我们再来看字符Char类型：

```
info: .\.\.\.\Test3\Basic.lean:57:0: structure Char : Type
number of parameters: 0
fields:
  Char.val : UInt32
  Char.valid : self.val.isValidChar
constructor:
  Char.mk (val : UInt32) (valid : val.isValidChar) : Char
```

Char类型是一个结构体，它有一个字段val，类型是UInt32。isValidChar是一个函数，用于判断val是否是一个有效的字符。

我们再来看UInt32类型：

```
info: .\.\.\.\Test3\Basic.lean:53:0: structure UInt32 : Type
number of parameters: 0
fields:
  UInt32.toBitVec : BitVec 32
constructor:
  UInt32.mk (toBitVec : BitVec 32) : UInt32
```

UInt32类型也是一个结构体，它有一个字段toBitVec，类型是BitVec 32。

注意，这是一个类型依赖。类型不是BitVec，而是依赖于后面的值的类型。用人话就说就是，不同长度的BitVec不是同一个类型。

```
info: .\.\.\.\Test3\Basic.lean:73:0: structure BitVec (w : Nat) : Type
number of parameters: 1
fields:
  BitVec.toFin : Fin (2 ^ w)
constructor:
  BitVec.ofFin {w : Nat} (toFin : Fin (2 ^ w)) : BitVec w
```

这也是我们明明要讲基本数据类型，但是上一节要讲类型依赖的原因。很多语言中非常基础的类型，在Lean中要么是归纳类型，要么是结构体，甚至有些还是类型依赖的类型。

同理，UInt8, UInt16, UInt64也都是依赖不同长度的BitVec。

```
info: .\.\.\.\Test3\Basic.lean:49:0: structure UInt8 : Type
number of parameters: 0
fields:
  UInt8.toBitVec : BitVec 8
constructor:
  UInt8.mk (toBitVec : BitVec 8) : UInt8
info: .\.\.\.\Test3\Basic.lean:51:0: structure UInt16 : Type
number of parameters: 0
fields:
  UInt16.toBitVec : BitVec 16
constructor:
  UInt16.mk (toBitVec : BitVec 16) : UInt16
info: .\.\.\.\Test3\Basic.lean:53:0: structure UInt32 : Type
number of parameters: 0
fields:
  UInt32.toBitVec : BitVec 32
constructor:
  UInt32.mk (toBitVec : BitVec 32) : UInt32
info: .\.\.\.\Test3\Basic.lean:55:0: structure UInt64 : Type
number of parameters: 0
fields:
  UInt64.toBitVec : BitVec 64
constructor:
  UInt64.mk (toBitVec : BitVec 64) : UInt64
```

与List是个归纳类型不同，Array是个结构体。

```
info: .\.\.\.\Test3\Basic.lean:61:0: structure Array.{u} (α : Type u) : Type u
number of parameters: 1
fields:
  Array.toList : List α
constructor:
  Array.mk.{u} {α : Type u} (toList : List α) : Array α
```

另外，Lean4中还有向量Vector，它继承自Array，但是依赖于长度，每一种长度的Vector是不同的类型。

```
info: .\.\.\.\Test3\Basic.lean:77:0: structure Vector.{u} (α : Type u) (n : Nat) : Type u
number of parameters: 2
parents:
  Vector.toArray : Array α
fields:
  Array.toList : List α
  Vector.size_toArray : self.size = n
constructor:
  Vector.mk.{u} {α : Type u} {n : Nat} (toArray : Array α) (size_toArray : toArray.size = n) : Vector α n
resolution order:
  Vector, Array
```

定长的向量和有限的有理数都是类型依赖的经典例子。

## 字符串

最后，字符串还需要单独拿出来说一下。

字符串是一个结构体，只有一个字段是字符的列表。

```
info: .\.\.\.\Test3\Basic.lean:63:0: structure String : Type
number of parameters: 0
fields:
  String.data : List Char
constructor:
  String.mk (data : List Char) : String
```

但是因为不定长字符编码的问题，字符串的长度远比字符列表的长度要复杂。

首先，如果你是在Windows系统上运行Lean 4，可能会遇到显示中文乱码的问题。这是因为Windows默认使用GBK编码，而Lean 4使用的是UTF-8编码。

我们可以通过修改控制台的代码页为 UTF-8（代码页 65001）来支持 UTF-8 字符串的输出。

```shell
chcp 65001
```

在 Lean 4 中，字符串是 Unicode 字符的序列。字符串字面量使用双引号括起来。我们仍然使用lenght函数来获取字符串的长度。

```lean
  let s10 := "数学世界666"
  IO.println s10
  let n10 := s10.length
  IO.println n10
```

运行结果如下：

```
数学世界666
7
```

我们知道，utf-8编码的中文字符占用超过一个字节，字符串长度取7并不是按照字符个数计算的。我们可以使用toUTF8函数来将其转换成字节数组，然后计算这个ByteArray的长度：

```lean
  let n11 := s10.toUTF8.size
  IO.println n11
```

我们可以看到，这个字符串占用了15个字节。

字符串的每个字符都是Unicode字符，所以我们可以使用toList函数将其转换成字符列表，然后使用get函数获取每个字符。

```lean
  let slist := s10.toList
  IO.println slist
  let c01 := slist[0]
  IO.println c01
  let c02 := slist[1]
  IO.println c02
  let c03 := slist[2]
  IO.println c03
```

输出如下：

```
[数, 学, 世, 界, 6, 6, 6] 
数                     
学                     
世                     
```

但是，如果你使用get函数获取字符串中的字符，你会发现它需要的是一个String.Pos类型。在起点的时候还好办，Lean4会自动推断出起点是0。

```lean
  let startPos : String.Pos := 0
  let c000 := s10.get startPos
  IO.println c000
```

但是，如果你想要获取下一个字符，就需要使用next函数来获取下一个位置：

```lean
  let nextPos : String.Pos := s10.next startPos
  let c001 := s10.get nextPos
  IO.println c001
  let nextPos2 : String.Pos := s10.next nextPos
  let c002 := s10.get nextPos2
  IO.println c002
```

输出如下：

```
数
学
世
```

## 小结

这一节我们学习了结构体，通过结构体，可以定义变量，也可以定义函数，还可以通过继承来扩展结构体。

同时，通过对于基本类型的分析，我们发现了归纳类型和类型依赖在Lean4中的重要性。
