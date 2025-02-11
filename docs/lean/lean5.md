# 面向程序员的Lean 4教程(5) - 从三段论说起

这一节我们复习一下离散数学中的命题逻辑。

如果读者大大没有学过命题逻辑也不要紧，我们从更简单的三段论说起。

三段论是古希腊哲学家亚里士多德提出的一种推理方法，它由三个命题组成：

1. 大前提：所有M都是P
2. 小前提：所有S都是M
3. 结论：所有S都是P

比如，著名的三段论：

1. 大前提：所有的人都会死
2. 小前提：苏格拉底是人
3. 结论：苏格拉底会死

我们再简化一下，就是著名的Modus Ponens：

1. 大前提：如果A，则B
2. 小前提：A
3. 结论：B

用数学语言表示就是：

1. 大前提：A → B
2. 小前提：A
3. 结论：B

我们可以用Lean4来证明，theorem用来定义定理，intro用来引入前提，apply用来应用前提。

```lean
theorem modus_ponens {A B : Prop} : (A → B) → A → B := by
  intro h1 h2  -- 分别引入前提h1（A→B）和h2（A）
  apply h1      -- 应用h1，目标变为证明A
  exact h2      -- 用h2完成证明
```

我们可以理解为Prop是命题，A → B 是命题A推出命题B。

## 基于分离规则的推演系统

modus ponens的常见翻译是分离规则。

我们就基于这一个规则来构建一个推演系统。

第一个定理：`A → (B → A)`

```lean
theorem proof1 {A B : Prop} : A → (B → A) := by
  intro a  -- 假设 `A` 为真，记作 `a : A`
  intro _  -- 假设 `B` 为真，记作 `b : B`（实际未使用，故用 `_` 忽略命名）
  exact a  -- 直接使用已知的 `a : A` 完成证明
```

可以看到，跟前面的Modus Ponens的证明思路很像。

第二个定理：`(A → (B → C)) → (A → B) → (A → C)`

```lean
theorem proof2 {A B C : Prop} : (A → (B → C)) → (A → B) → (A → C) := by
  -- 引入假设 `h₁ : A → (B → C)` 和 `h₂ : A → B`
  intro h₁ h₂ h₃
  -- 组合 `h₁ h₃`（应用 `A → (B → C)` 到 `h₃ : A`）得到 `B → C`
  -- 再组合 `h₂ h₃`（应用 `A → B` 到 `h₃ : A`）得到 `B`
  exact h₁ h₃ (h₂ h₃)
```

第三个定理：`(¬B → ¬A) → (¬B → A) → B `

```Lean
import Mathlib.Tactic.ByContra

theorem proof3 (A B : Prop) : (¬B → ¬A) → (¬B → A) → B := by
  intro h1 h2
  by_contra h3       -- 反证法：假设 ¬B
  have h4 := h1 h3   -- 得到 ¬A
  have h5 := h2 h3   -- 得到 A
  contradiction      -- 矛盾（A 和 ¬A）
```

之所以要引入`Mathlib.Tactic.ByContra`，是因为`by_contra`是Lean 4中的一个策略，用于反证法。

Lean4默认使用直觉主义逻辑，直觉主义逻辑不承认排中律，即`A \/ \not A`不一定成立。

直觉逻辑可能有的同学没有学过，其实在2001年版的北京大学出版社的《离散数学》教材中，王捍贫老师在仅有四章的篇幅中，最后一章专门就讲的直觉逻辑。

不过我们不着急，趁着带领大家熟悉Lean4的机会，我们也顺路把离散数学复习一下。

## 命题演算的自然推理

按照逻辑书的惯例，Γ ⊢ a 表示a在Γ下可证。

我们下面来看如何用Lean4来证明命题演算的自然推理。

比如我们先看`->`消去律: 如果Γ ⊢ a → b，且 Γ ⊢ a 则 Γ ⊢ b。

这里我们用`example`来证明，`example`可以理解为匿名定理。

```lean
example (h₁ : a → b) (h₂ :  a) :  b := by
  -- 应用蕴含式(a → b)的证明h₁到a的证明h₂，直接得到b的证明。
  apply h₁
  exact h₂
```

下面我们做个小练习：{a → b, b → c, a} ⊢ a → c

```lean
example (h₁ : a → b) (h₂ : b → c) : a → c := by
  intro a'  -- 假设a'，目标变为证明c
  apply h₂  -- 应用b → c，目标变为证明b
  apply h₁  -- 应用a → b，目标变为证明a
  exact a'  -- 使用假设a'完成证明
```

这种证明不够函数式，我们可以通过函数组合的方式来证明：

```lean
example (h₁ : a → b) (h₂ : b → c) : a → c := by
  exact λ a' => h₂ (h₁ a')  -- 直接组合h₁和h₂，生成a → c的证明
```

小练习2：{a → b, b → c, a} ⊢ c

```lean
example (h1 : a → b) (h2 : b → c) (h3 : a) : c := by
  apply h2  -- 目标变为证明b
  apply h1  -- 目标变为证明a

  exact h3  -- 直接使用前提a完成证明
```

也可以写成函数式：

```lean
example (h1 : a → b) (h2 : b → c) (h3 : a) : c := by
  -- 通过两次Modus Ponens组合得到c
  exact h2 (h1 h3)
```

再来一个例子：若Γ, a ⊢ b 且 Γ, a ⊢ ¬b，则 Γ ⊢ ¬a

```lean
example (h₁ : (a → b)) (h₂ : (a → ¬b)) : (¬a) := by
  intro ha  -- 假设a，目标变为推导出False
  exfalso   -- 将目标转换为False
  apply h₂ ha  -- 应用Γ, a ⊢ ¬b，得到¬b（即b → False）
  apply h₁ ha  -- 应用Γ, a ⊢ b，得到b
```

## 引入否定假设

下面我们证明一下` (a→b)⊢(¬b→¬a)`

我们用三种证法来分别证明：

```lean
theorem contrapositive₂ {a b : Prop} (h : a → b) : ¬b → ¬a := by
  intro hnb         -- 引入假设 ¬b
  intro ha          -- 引入假设 a
  apply hnb         -- 应用 ¬b（目标变为证明 b）
  apply h           -- 应用 a → b（目标变为证明 a）
  exact ha          -- 使用假设 a 完成证明
```

从中我们可以看到，假设a为真，就是ha，假设b为假，就是hnb。

我们再换用反证法，其实有了hnb之后，反证法就是同样的策略了：

```lean
theorem contrapositive₃ {a b : Prop} (h : a → b) : ¬b → ¬a := by
  intro hnb         -- 引入假设 ¬b
  by_contra ha      -- 反证法：假设 a 成立
  apply hnb         -- 应用 ¬b（目标变为证明 b）
  apply h           -- 应用 a → b
  exact ha          -- 使用反证法中的 a 假设
```

最后我们还是用函数式来写：

```lean
theorem contrapositive₁ {a b : Prop} (h : a → b) : ¬b → ¬a :=
  fun hnb : ¬b =>   -- 假设 ¬b 成立
  fun ha : a =>     -- 假设 a 成立
  absurd (h ha) hnb -- 通过 h 得到 b，与 ¬b 矛盾
```

我们用#print命令来打印一下：

```lean
#print contrapositive₁
#print contrapositive₂
#print contrapositive₃
```

可以看到，其实方法2和3是等价的：

```
info: .\.\.\.\Test3\Logic2.lean:84:0: theorem contrapositive₁ : ∀ {a b : Prop}, (a → b) → ¬b → ¬a :=
fun {a b} h hnb ha => absurd (h ha) hnb

info: .\.\.\.\Test3\Logic2.lean:86:0: theorem contrapositive₂ : ∀ {a b : Prop}, (a → b) → ¬b → ¬a :=
fun {a b} h hnb ha => hnb (h ha)
info: .\.\.\.\Test3\Logic2.lean:88:0: theorem contrapositive₃ : ∀ {a b : Prop}, (a → b) → ¬b → ¬a :=
fun {a b} h hnb ha => hnb (h ha)
```

## 小结

通过使用Lean4，之前离散数学或者是数理逻辑中印刷的东西就活生生地在代码中跑起来了。这将大大有助于我们学习数理逻辑，借着学习Lean4的机会，大大好好复习下数理逻辑吧。

