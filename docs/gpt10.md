# 2023年的深度学习入门指南(10) - 思维链

有很多同学因为惯性思维，认为只有编程改代码，或者是调整模型的超参数，哪怕是调用脚本进行微调训练，才是正确使用大模型的方法，而提示工程是骗子。

我们前面讲了，大模型有的能力是涌现出来的，其实我们并不了解其背后的原理。但是不懂原理并不等于它不存在。

既然少样本学习的能力在大于GPT3的模型上已经涌现出来了，那么与其去怀疑它，还不如去想办法去更好地引导大模型。这些引导激活大模型的研究，也基本上都出自于大模型的研究团队，比如谷歌大脑团队、马里亚大学、韩国科学技术研究院等。

## 随规模产生的新能力

![model scale](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/model_scale1.png)

从上图我们可以看到，不管是Google的LaMDA，还是openai的GPT#，还是DeepMind的Chinchilla等在10的22次方次FLops训练之前，处理数学问题就跟瞎猜没啥区别。

一旦突破了一个临界点之后，模型就突然开始突飞猛进。例如，GPT-3论文显示，语言模型执行多位数字加法的能力在从1亿到130亿参数的模型中具有平坦的缩放曲线（大约是随机表现），在此时性能大幅提升。

论文作者认为，这是一种不可预测的现象。没有办法在小模型预测出更大模型能出现什么能力。这种新兴能力的存在引发了一个问题，即是否可以通过进一步扩展语言模型的规模来进一步扩展语言模型的能力范围。为什么扩大规模会解锁新兴能力？由于计算资源昂贵，是否可以通过其他方法解锁新兴能力而不增加规模（例如更好的模型架构或训练技术）？当特定能力出现时，是否会解锁语言模型的新实际应用？

做更大的模型可能跟我们无关。我们先学习已经发现的一种新能力。

一种确认的新能力就是去年1月提出，到今年1月已经更新到第6版的《Chain-of-Thought Prompting Elicits Reasoning in Large Language Models》，它提出的概念叫做思维链。

## 思维链

所谓思维链，是指生成一系列中间推理步骤，用以辅助大模型去进行推理的方法。

![](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/chainoft1.png)

正如大家所知，大模型其实就是一个语言模型，它的强项是在不断增加语料之后，涌现出了思考能力。但是，有些能力没有因模型的简单变大而有显著提高，比如数字计算、逻辑推理等就是人类做起来也是比较费脑子的事情。

而且，随着思维链的研究，各大模型对于推理方面的进步是肉眼可见的。

比如说，论文中举的失败的例子：
![图1](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/CoT1.png)

我们用chatgpt试试：
![chatgpt](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/chatgpt2.png)

看看人家chatgpt，分步思考清晰着呢。

![claude](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/claude1.png)

Claude主动声明不擅长解决数学问题，但是仍然可以算对这个题目。而且分布很清晰。

![Baidu](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/bidu1.png)

百度文心一言虽然没有写过程，但是结果也是对的。

思维链的优点主要有三方面：
- 第一，思维链有助于提示大模型分配更多的资源来处理复杂的问题
- 第二，思维链有利于提升黑盒大模型的可解释性。也间接降低了胡说八道的可能性
- 第三，思维链可以在不改变现有大模型能力的情况下提升效果

### 数学题

第1题，输入从1到500，一共需要敲击键盘多少下。

![](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/math1.png)

Chatgpt这个分步分析得不错。

![](https://xulun-mooc.oss-cn-beijing.aliyuncs.com/math1-c.png)

Claude也是同样分析的不错。

New bing还引用了数学网站的信息，好像是抄到答案了：）

```
This seems to be a math problem. I can try to help you solve it using the information from the web. According to ¹²³, while typing numbers from 1 to 1000, there are **9** single digit numbers: from 1 to 9. Each of these numbers requires **one** keystroke. That is **9** key strokes. There are **90** two-digit numbers: from 10 to 99. Each of these numbers requires **2** keystrokes. Therefore, **180** keystrokes to type the 2-digit numbers. There are **900** three-digit numbers: from 100 to 999. Each of these numbers requires **3** keystrokes. Therefore, **2700** keystrokes to type the 3-digit numbers. 1000 is a four-digit number which requires **4** keystrokes.

To type numbers from 1 to 500, we can use the same logic but only consider the numbers up to 500. There are still **9** single digit numbers and **90** two-digit numbers, so they require the same number of keystrokes as before: **9 + 180 = 189**. There are only **400** three-digit numbers: from 100 to 499. Each of these numbers requires **3** keystrokes. Therefore, **1200** keystrokes to type the 3-digit numbers. 500 is a three-digit number which requires **3** keystrokes.

Adding up all the keystrokes, we get **189 + 1200 + 3 = 1392**. Therefore, the answer choice is (b) **1392**.

源: 与必应的对话， 2023/4/28
(1) How many keystrokes are needed to type numbers from 1 to 1000?. https://gmatclub.com/forum/how-many-keystrokes-are-needed-to-type-numbers-from-1-to-215079.html.
(2) How many keystrokes are needed to type numbers from 1 to 1000? - BYJU'S. https://byjus.com/question-answer/how-many-keystrokes-are-needed-to-type-numbers-from-1-to-1000/.
(3) How many keystrokes are needed to type numbers from 1 to 1000 on a .... https://abhipedia.abhimanu.com/Article/multiquant/MTQxOTAEEQQVV/How-many-keystrokes-are-needed-to-type-numbers-from-1-to-1000-on-a-standard-keyboard-Quant-MASTER-.
```

