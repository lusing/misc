# 面向程序员的Lean 4教程(3) - 作为一等公民的类型

有了前面像传统编程语言一样使用Lean 4的经验后，我们引入一点Lean 4中与传统语言不太一样的东西：类型。

比如，在Lean 4中，类型如Bool, Nat, Int本身也是一级对象，可以绑定到变量中，它们的类型是Type.

## Type类型

我们来看将类型做为一等公民的用法：

```lean
def t1 : Type := Nat
def t3 : Type := Bool
def t4 : Type := UInt16
```

除了基本类型是Type类型，我们还可以看到下面的类型：

```lean
def t2 : Type := Nat → Nat
def t8 := Nat → Nat → Nat
```

我们可以使用`#check`命令来查看一个变量的类型：

```lean
#check t1

#check t2

#check t3

#check t4

#check t8
```

输出如下：

```
info: .\.\.\.\Test3\Basic.lean:21:0: t1 : Type
info: .\.\.\.\Test3\Basic.lean:23:0: t2 : Type
info: .\.\.\.\Test3\Basic.lean:25:0: t3 : Type
info: .\.\.\.\Test3\Basic.lean:27:0: t4 : Type
info: .\.\.\.\Test3\Basic.lean:35:0: t8 : Type
```

聪明的你可能会想到，既然Type类的对象是一等公民，那么Type本身是不是也是一等公民呢？

恭喜你，你发现了Girard 悖论，类似于集合论中的罗素悖论，一个集合不能包含自身，一个类型也不能是超越自己的类型。

在Lean 4中，通过类型的层次来解决这个问题。

Type的变量的类型是Type 1:

```lean
def t5 : Type 1:= Type
```

同样，Type 1变量的类型是Type 2：

```lean
def t6: Type 2 := Type 1
```

同理，`Type 1 → Type`的类型也是Type 2:

```lean
def t7: Type 2 := Type 1 → Type
```

打印出来看看：

```lean
#check t5

#check t6

#check t7
```

结果如下：

```
info: .\.\.\.\Test3\Basic.lean:29:0: t5 : Type 1
info: .\.\.\.\Test3\Basic.lean:31:0: t6 : Type 2
info: .\.\.\.\Test3\Basic.lean:33:0: t7 : Type 2
```

## 函数的类型

同类型一样，Lean 4中的函数也是一等公民。那么，函数是什么类型呢？

我们看下下面4个函数：

```lean
def f1 := fun x => x+(1:Int)

def f2 := λ x => x+(2:Int)

def f3 := (. + (3:Int))

def f4 (x : Int) : Int :=
  x+4
```

其中，最后的f4是传统语言最熟悉的函数。
f1也是现代语言中普遍支持的lambda函数的常见表达形式。f2不伪装了，直接就写λ表达式。

我们检查下它们的类型：

```lean
#check f1

#check f2

#check f3

#check f4
```

运行结果如下：

```
info: .\.\.\.\Test3\Basic.lean:88:0: f1 (x : Int) : Int
info: .\.\.\.\Test3\Basic.lean:90:0: f2 (x : Int) : Int
info: .\.\.\.\Test3\Basic.lean:92:0: f3 : Int → Int
info: .\.\.\.\Test3\Basic.lean:94:0: f4 (x : Int) : Int
```

其实它们的类型都是`Int → Int`, 它也是Type类的实例。

## 归纳类型

类似于古老的枚举类型，归纳类型在现代语言中也越来越流行。

归纳类型是将类型中所有可能的值都一一列举出来，最简单的就像布尔类型，比如我们定义一个Bool2类型：

```lean
inductive Bool2 : Type where
  | true2 : Bool2
  | false2 : Bool2

def b1 : Bool2 := Bool2.true2
def b2 : Bool2 := Bool2.false2

#check b1
#check b2
```

我们可以给Bool2写一个取反函数：

```lean
def not2 : Bool2 → Bool2
  | Bool2.true2 => Bool2.false2
  | Bool2.false2 => Bool2.true2
```

这个函数我们直接取`Bool2 → Bool2`作为类型。

或者我们懒得写类型，就直接写$\lambda$表达式，类型让Lean4自己去推断：

```lean
def not2 := λ
  | Bool2.true2 => Bool2.false2
  | Bool2.false2 => Bool2.true2

#check not2
```

## 基本类型的真面目

这一节我们学习一个新的命令`#print`, 通过它我们可以查看到之前学习的类型的真面目。

比如我们看看Bool类型是个什么：

```lean
#print Bool
```

输出如下：

```
info: .\.\.\.\Test3\Basic.lean:41:0: inductive Bool : Type
number of parameters: 0
constructors:
Bool.false : Bool
Bool.true : Bool
```

原来Bool类型就是由false和true归纳出来的类型。

我们再看看Nat是个什么：

```
info: .\.\.\.\Test3\Basic.lean:39:0: inductive Nat : Type
number of parameters: 0
constructors:
Nat.zero : Nat
Nat.succ : Nat → Nat
```

原来Nat也是个归纳类型，由0和后继函数两者组成。

我们来用归纳类的方式来使用Nat:

```lean
def chkNat : IO Unit := do
  let n1 := Nat.zero
  let n2 := n1.succ
  IO.println n1
  IO.println n2
```

输出为0和1.

我们再看看Int:

```
info: .\.\.\.\Test3\Basic.lean:43:0: inductive Int : Type
number of parameters: 0
constructors:
Int.ofNat : Nat → Int
Int.negSucc : Nat → Int
```

原来堂堂Int类型是两个自然数到整数的转换函数的归纳。

我们来操练一下：

```lean
def chkInt : IO Unit := do
  let i1 : Int := Int.ofNat Nat.zero.succ
  let i2 : Int := Int.negSucc Nat.zero
  IO.println i1
  IO.println i2
```

输出为1和-1。

再来看一个复杂的一点的，List:

```lean
#print List
```

输出如下：

```
info: .\.\.\.\Test3\Basic.lean:57:0: inductive List.{u} : Type u → Type u
number of parameters: 1
constructors:
List.nil : {α : Type u} → List α
List.cons : {α : Type u} → α → List α → List α
```

类似于自然数的定义，列表由空列表和拼接列表的函数两个部分组成。

列表需要指定类型，这在普遍支持泛型的时代也不是什么新鲜事。

我们用这两个基本函数来构造列表：

```lean
def chkList : IO Unit := do
  let l1 : List Nat := List.nil
  let l2 : List Nat :=List.cons 1 l1
  let l3 : List Nat :=List.cons 2 l2
  IO.println l1
  IO.println l2
  IO.println l3
```

输出如下：

```
[]
[1]
[2, 1]
```

## 类型依赖于变量

如果类型只依赖于变量的类型，就像其他语言中的泛型，大家都很熟悉。

如果类型依赖于变量的值，那么对于很多同学来说，可能就是超出预期了。

其实，因为类型是一等公民，它们其实都是Type类的实例，所以简单的根据变量的值来决定类型，只是改变Type类的实例的值而已。

比如我们有一个类型，依赖的变量如果是0，那么类型是Int；其他情况下类型是String。

```lean

-- 定义一个依赖类型，根据变量的值决定类型
def TypeDependingOnValue (n : Nat) : Type :=
  if n = 0 then Int else String

-- 示例使用
def example1 : TypeDependingOnValue 0 := (0 : Int)  -- 类型为 Int
def example2 : TypeDependingOnValue 1 := "Hello"   -- 类型为 String

-- 打印结果
#eval example1  -- 输出: 0
#eval example2  -- 输出: "Hello"

```

这样看起来跟其他语言的代码也没有什么本质区别，是吧？

下面我们再把上面的代码改的更实用一点。比如你现在当助教，要录入学生的考试成绩。如果缺考的话，写成0分或者-1分都不优雅。我们可以根据考生的考试状态来决定分数的类型，就跟我们上面的例子类似。参加考试的就录分数，就是Nat类型，缺考的就录成String类型。

```lean
-- 定义学生是否参加考试的类型
inductive ExamStatus
| attended
| notAttended


-- 定义分数类型，依赖于学生是否参加考试
def Score (status : ExamStatus) : Type :=
  match status with
  | ExamStatus.attended => Nat
  | ExamStatus.notAttended => String

-- 定义依赖对类型，表示学生及其分数
def StudentWithScore : Type := Σ (status : ExamStatus), Score status

-- 定义一个学生参加了考试，分数为85
def student1 : StudentWithScore := ⟨ExamStatus.attended, (85:Nat)⟩
#check student1
#print student1

-- 定义一个学生没有参加考试，分数为"未参加"
def student2 : StudentWithScore := ⟨ExamStatus.notAttended, "未参加"⟩
#check student2
#print student2
```

我们再来一个更高级的，我们想打印变量的值，但是需要这个变量支持ToString实例。

我们可以写出这样一个东西：

```lean
def examplePi : ∀ (α : Type) [inst : ToString α], α → String
| _, _, x => s!"The input is: {x}"

#check examplePi
#print examplePi
```

如果这种看起来不太习惯，我们可以换成另一个写法：

```lean
def toStringPi : ∀ (α : Type) [inst : ToString α], α → String :=
  λ α inst x => s!"The input is: {x}"

#check toStringPi
#print toStringPi
```

最后这一小节暂时还不理解也没关系，我们只是举例说明类型依赖可以做很多传统泛型做不到的事情。

## 小练习

用归纳类型定义一个五行的类，成员是木火土金水。

然后写两个函数，输出它的相生相克。

比如，木生火，火生土，土生金，金生水，水生木。
水克火，火克金，金克木，木克土，土克水。

最后写几个测试用例，验证一下。

参考例子：

```lean
-- 定义五行的类型
inductive WuXing : Type
  | metal  -- 金
  | wood   -- 木
  | water  -- 水
  | fire   -- 火
  | earth  -- 土

open WuXing

-- 定义相生关系
def generates : WuXing → WuXing → Prop
  | water, wood   => true  -- 水生木
  | wood,  fire   => true  -- 木生火
  | fire,  earth  => true  -- 火生土
  | earth, metal  => true  -- 土生金
  | metal, water  => true  -- 金生水
  | _,     _      => false

-- 定义相克关系
def overcomes : WuXing → WuXing → Prop
  | water, fire   => true  -- 水克火
  | fire,  metal  => true  -- 火克金
  | metal, wood   => true  -- 金克木
  | wood,  earth  => true  -- 木克土
  | earth, water  => true  -- 土克水
  | _,     _      => false

-- 示例：检查五行之间的相生和相克关系
example : generates water wood := by simp [generates]
example : overcomes water fire := by simp [overcomes]
example : ¬ generates water fire := by simp [generates]
example : ¬ overcomes water wood := by simp [overcomes]
```

虽然用了一些没讲到的语法，但是这个例子应该不难理解，就像写测试用例一样，没有什么复杂的。

## 小结

这一节我们学习了类型作为一等公民的用法，包括类型依赖于变量，类型依赖于值，类型依赖于函数等。

我们没有讲$\Pi$类型依赖和$\Sigma$类型依赖之类的理论，我们也没有讲数学证明。我们只要理解类型是一等公民就可以了。


