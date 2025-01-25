# 面向程序员的Lean4教程(2) - 数组和列表

上一节我们介绍了Lean4的基本语法，因为大部分程序员都有一定的编程基础，所以并没有介绍过细。这一节我们介绍Lean4中的线性表结构：数组和列表，顺带复习一下上一节的流程控制等内容。

## 数组

### 创建数组

Lean4中的数组可以用#[]来进行初始化，也可以用`Array.mk`来创建。数组的元素类型可以是任意类型，但是所有元素的类型必须相同。

```lean
  let a1 : Array Nat := #[1, 2, 3, 4, 5]
  let a2 : Array Int := Array.mk [1, 2, 3, 4, 5]
  IO.println a1
  IO.println a2
```

输出如下：

```
#[1, 2, 3, 4, 5]
#[1, 2, 3, 4, 5]
```

### 访问数组元素

Lean4中的数组元素可以通过`a[i]`来访问，其中`a`是数组，`i`是索引。

```lean
  IO.println a1[0]
  IO.println a2[1]
```

我们也可以通过Array.get来访问数组元素。

如果想直接获取值，我们可以使用`get!`，如果可能为空，我们要使用`get?`。

```lean
  IO.println a1.get! 0
  IO.println a2.get? 1
```

get?返回一个Option类型，我们可以通过`match`来处理。

```lean
  match a1.get? 10 with
  | none => IO.println "Not found"
  | some v => IO.println v
```

### 修改数组元素

Lean4中的数组元素是只读的，我们不能直接修改数组元素。如果想修改数组元素，我们可以使用`Array.set!`，它会返回一个新的数组。

```lean
  let a4 : Array Nat := Array.set! a1 0 10
  IO.println a4
```

### 获取数组长度

Lean4中的数组长度可以通过Array.size来获取：

```lean
  IO.println (Array.size a4)
  IO.println (a4.size)
```

### 拼接数组

Lean4中的数组可以通过`Array.append`来拼接：

```lean
  let a5 := Array.append a1 a4
  IO.println a5
```

输出如下：

```
#[1, 2, 3, 4, 5, 10, 2, 3, 4, 5]
```

### 数组切片

Lean4中的数组可以通过`Array.extract`来切片：

```lean
  let a6 := Array.extract a5 1 4
  IO.println a6
```

输出如下：

```
#[2, 3, 4]
```

### 数组反转

Lean4中的数组可以通过`Array.reverse`来反转：

```lean
  let a7 := Array.reverse a5
  IO.println a7
```

输出如下：

```
#[5, 4, 3, 2, 10, 5, 4, 3, 2, 1]
```

### 数组排序

Lean4中的数组可以通过`Array.qsort`来排序：

```lean
  let a8 := Array.qsort a5 (fun x y => x < y)
  IO.println a8
```

输出如下：

```
#[1, 2, 2, 3, 3, 4, 4, 5, 5, 10]
```

fun定义了一个匿名函数，x < y是函数体。

但是这样的写法不高级，在Lean4中我们可以使用洞来简化：

```lean
  let a8 := Array.qsort a5 (. < .)
  IO.println a8
```

fun这样的关键字省了，形式参数用`.`来表示，只突出了最重要关系判断的部分。是不是很高级？：）

### 查找数组是否包含某个元素

Lean4中的数组可以通过`Array.contains`来查找是否包含某个元素：

```lean
  let b2 := Array.contains a8 10
  IO.println b2
```

输出为`true`。

### 查找符合条件的元素

如果只想找到一个符合条件的元素，我们可以使用`Array.find?`：

```lean
  let a10 := Array.find? (. > 3) a8
  IO.println a10
```

输出为`some 4`。

如果想找到所有符合条件的元素，我们可以使用`Array.filter`：

```lean
  let a11 := Array.filter (. > 3) a8
  IO.println a11
```

输出为`#[4, 4, 5, 5, 10]`。

### 像栈一样操作数组

Lean4中的数组可以通过`Array.push`来像栈一样操作：

```lean
  let a11 := a10.push 100
  IO.println a11
```

输出如下：

```
#[8, 10, 2, 4, 6, 8, 10, 12, 14, 16, 18, 100]
```

然后我们可以通过`Array.pop`来弹出栈顶元素，这样数组就变成了原来的数组：

```lean
  let a12 := a11.pop
  IO.println a12
```

输出如下：

```
#[8, 10, 2, 4, 6, 8, 10, 12, 14, 16, 18]
```

## 列表

数组一般是在内存中连续存储的，而列表是在内存中不连续存储的。

可以通过`[]`来创建列表：

```lean
  let l1 : List Nat := [1, 2, 3, 4, 5]
  IO.println l1
```

输出如下：

```
[1, 2, 3, 4, 5]
```

除此之外，我们还可以使用List.range来创建：

```lean
  let l2 := List.range 10
  IO.println l2
```

输出如下：

```
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### 列表的连接

Lean4中的列表可以通过`List.append`来连接，但是更常用的是`++`运算符：

```lean
  let l3 := l1 ++ l2
  IO.println l3
```

输出如下：

```
[1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### 获取列表长度

Lean4中的列表长度可以通过List.length来获取：

```lean
  IO.println (List.length l3)
  IO.println (l3.length)
```

输出如下：

```
15
15
```

### 列表的反转

同数组一样，Lean4中的列表可以通过`List.reverse`来反转：

```lean
  let l4 := l3.reverse
  IO.println l4
```

输出如下：

```
[9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1]
```

### 列表的遍历

按命令式编程的思维，传统方式就是用for循环来遍历列表：

```lean
  for x in l5 do
    IO.println x
```

用函数式编程的方式，我们可以使用`map`来遍历列表。比如生成一个新的列表，每个元素都是原来的元素的2倍：

```lean
  let l5 := l3.map ( . * 2)
  IO.println l5
```

### 列表的切片

Lean4中的列表可以通过`List.take`来获取前n个元素：

```lean
  let l6 := l5.take 3
  IO.println l6
```

输出如下：

```
[2, 4, 6]
```

Lean4中的列表可以通过`List.drop`来删除前n个元素：

```lean
  let l7 := l5.drop 3
  IO.println l7
```

输出如下：

```
[8, 10, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
```

### 列表的折叠

折叠是函数式编程中的一个重要概念，它可以将一个列表中的元素通过某种规则合并成一个值。

最常用的折叠情况就是求和：

```lean
  let l8 := l7.foldl (. + .) 0
  IO.println l8
```

输出为`108`。

我们同样可以使用`foldr`来从右边开始折叠，这次我们求积。但是我们的列表里有0，所以我们先用filter过滤掉0：

```lean
  let l9 := l7.filter (. > 0)
  IO.println (l9.foldr (. * .) 1)
```

### 列表转换成数组

Lean4中的列表可以通过`List.toArray`来转换成数组：

```lean4
  let a10 := l9.toArray
  IO.println a10
```

输出如下：

```
#[8, 10, 2, 4, 6, 8, 10, 12, 14, 16, 18]
```

### 列表去重

Lean4中的列表可以通过`List.eraseDup`来去重，也就是去掉重复的元素：

```lean
  let l10 := l5.eraseDups
  IO.println l10
```

输出如下：

```
[2, 4, 6, 8, 10, 0, 12, 14, 16, 18]
```

### 列表的量词

Lean4中的列表可以通过`List.all`来判断是否所有元素都满足某个条件，例：

```lean
  let b3 := l5.all (. > 0)
  IO.println b3
```

我们知道，l5中含有元素0，所以输出为`false`。

Lean4中的列表还可以通过`List.any`来判断是否有元素满足某个条件：

```lean
  let b4 := l5.any (. > 0)
  IO.println b4
```

因为除了0以外，l5中的所有元素都大于0，所以满足条件，输出为`true`。

### 给列表的头部添加元素

可以使用构造符号`::`来给列表的头部添加元素，当然，是生成新的列表：

```lean
  let l11 := 100 :: l5
  IO.println l11
```

输出如下：

```
[100, 2, 4, 6, 8, 10, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
```

在头部添加元素的时间复杂度是O(1)。

在尾部添加元素的时间复杂度是O(n)，就是我们之前介绍的`++`运算符。

## 可变的数组和列表

Lean4中的数组和列表是不可变的，如果想要可变的数组和列表，我们可以使用`IO.Ref`。
注意，从`IO.Ref`获取值时，我们需要使用`←`，不要写成`:=`。

```lean
    let mut r1 ← IO.mkRef #[1, 2, 3, 4, 5]
    let mut r2 ← IO.mkRef [1, 2, 3, 4, 5]
```

然后我们可以通过`IO.Ref.get`来获取值，通过`IO.Ref.set`来设置值。

```lean
  let mut r1 ← IO.mkRef #[1, 2, 3, 4, 5]
  let arr1 ← r1.get
  IO.println arr1
  r1.set (arr1.push 6)
  IO.println (← r1.get)
```

输出如下：

```
#[1, 2, 3, 4, 5]
#[1, 2, 3, 4, 5, 6]
```

对于列表，我们也是同样做法：

```lean
  let mut r2 ← IO.mkRef [1, 2, 3, 4, 5]
  let arr2 ← r2.get
  IO.println arr2
  r2.set (arr2 ++ [6])
  IO.println (← r2.get)
```

输出如下：

```
[1, 2, 3, 4, 5]
[1, 2, 3, 4, 5, 6]
```

## 小练习

下面我们做几个小练习，看看大家有没有适应Lean4的编程方式。

1. 将一个Nat类型的列表转换成Int类型的列表

例：

```lean
  let l13: List Nat := [1, 2, 3, 4, 5]
  let l14 := l13.map Int.ofNat
  IO.println l14
```

Int.ofNat是将Nat类型转换成Int类型的函数，所以我们不需要再定义一个新的函数，直接调用它就可以了。

2. 由于List只能顺序访问，我们将其转化成数组，然后排序，最后再转化回List。

例：

```lean
def sortList (lst : List Nat) : List Nat :=
  let arr := List.toArray lst  -- 将列表转换为数组
  let sortedArr := Array.qsort arr (. < .)   -- 对数组进行排序
  Array.toList sortedArr  -- 将排序后的数组转换回列表
```

3. 实现一个函数，遍历二维数组

最简单的方法就是使用两个for循环：

```lean
def traverse2DArray (arr : Array (Array Nat)) : IO Unit := do
  for row in arr do
    for elem in row do
      IO.print s!"{elem} "
    IO.println ""
```

当然也可以采用别的函数式的方法，或者递归的方法。

## 小结

本节我们不厌其烦地介绍了很多数组和列表的操作，在让大家熟悉这两种数据结构的同时，也是让大家进一步熟悉Lean4的基本编程方式。我们后面会继续深入，现在大家可以先练习一下用Lean4来写一些简单的代码。
