# Rust语言教程(1) - 一门没有GC的语言

## 缘起

本来这一系列文章并不在计划中。昨天跟一些同事讨论没有GC管理内存的问题。
讨论到没有GC情况下管理内存的学习曲线，有同事认为学习曲线不陡而是使用曲线陡。诚然，如果只有malloc和free，确实还是学习容易使用难的。到了C++引用了new和delete之后，学习曲线也还算是平和的，因为后面还有auto_ptr, 自动引用计数，右值引用和std::move等一大堆要学习的慢慢地增加进来。多线程的情况下，还会有更复杂一些的问题。
但是这比起Rust语言来，学习曲线都要好一些，因为大不了是memory leak，起码还可以编译通过。而在Rust语言中，初始的这个小陡坡如果踏不过去的话，可能会连编译都编不过。

这就是Rust的设计原则，不希望有一个比较重的运行时，通过编译时的勤奋来减少运行时的麻烦。

## Rust语言简介

Rust是Mozilla推出的，希望能够在多核时代充分发挥硬件能力的语言。Rust能够引起大家的关注，最初的原因是它的设计者之一的Brendan Eich是著名的JavaScript语言之父。不管程序员对于JavaScript有多少的诟病，JavaScript的成功是个难以重制的传奇。JavaScript在OOP如日中天的年代将self的原型链和scheme的函数式编程思想结合起来，显示出了Brendan Eich的功力。但是，JavaScript的最初版本的开发时间太短，没有充分给Brendan Eich发挥自己能力的机会。后来，有了历史包袱之后，就不是想怎么改就怎么改的情况了。JavaScript成为ECMA Script之后，4.0版本的被废弃掉，就是这个挫折的表现。
Rust是门新语言，三位主要创造人可以放开手脚去发挥。他们也确实是这么做的，语法不停改来改去。比如Rust最初是支持Go一样的轻量级多任务支持的，后来就被砍掉了，现在就是传统的线程之间像Go一样在来回发消息。
不过，其实Rust最早的创始人是Graydon Hoare. 他在2006年就开始开发了Rust，最早的编译器用OCaml来实现的。2009年，Hoare在Mozilla公司的支持下继续开发Rust。
2015年5月16日，Rust正式发布1.0版本。Hoare虽然离开了Mozilla，但是Rust在Mozilla的支持下在继续发展。

用Rust语言开发的Servo是一个类似Webkit的浏览器引擎。

Rust语言的一个重要特点就是没有一个厚重的运行时，所以也就没有GC这样的运行时的设施。坏处是我们得花精力学习在无GC情况下的设计，就是我们这一讲想做到的事情。好处是，轻量，可以用于系统编程，代码可以跑在裸机上，而不是一套运行时环境上。这样，理论上，Rust编译的代码可以像C和C+\+语言一样轻，可以用于嵌入式系统级软件开发，比如IoT设备上。做为对比的是Go语言，它有一个运行时，写应用的话Go很可能会胜出，但是在写系统级软件时，这个运行时环境可能就不是最理想的基础设施了。
举个小例子，比如Go和Rust同样来为Java写JNI代码。Rust就可以被当成C/C+\+用就是了，但是Go就会引入一个很好玩的问题，Java有GC，Go也有GC，这样的JNI代码如何处理二者间引用对象的问题？
Rust和C+\+之间基本可以无缝调用，反正本质上的思想也差不多。
Rust支持函数式的思想，支持C+\+所没有的类似于函数语言的宏。Rust支持泛型，支持traits，这是C++程序员所喜欢，而Java这样只是个语法糖方式所无法想象的。

常与Rust和Go一起比较的，还有Apple的swift语言和Digital Mars的D语言。它们的共同特点是高生产力的语法和全编译带来的高效率。

好啦，我们也先来一个Hello,World程序跟Rust语言打个招呼吧.
首先去官网下载Rust的包：https://www.rust-lang.org
在mac上，就用brew install rust就好了。
IDE的话，我用的是Visual Studio Code,大家可以自由选择。

```rust
fn main(){
	let s = "Hello,Rust!";
	println!("{}",s);
}
```

调用rustc，或者是通过cargo，就可以编译运行啦。
可以通过cargo init去创建一个工程，之后用cargo build或者cargo run就好了。
println!是一个输出用的宏，不要忘了后面那个提示它是个宏的叹号哟。

## Rust是一门没有GC的语言

好吧，闲话少说，我们开始面对学习一门类似于C++的，没有GC的新型语言的第一道关口，管理内存吧。

Rust同C++一样，存储空间既有堆，也有栈。对象放在栈上，可以复制，也可以随着退栈而被自动销毁掉。在C引入malloc之前，要么是静态分配，要么是在栈上分配，并没有复杂的堆上内存泄漏的难题。而以Java为代表的一类语言，默认都是在堆上分配对象，所以需要GC来实现堆上的管理。

我们也以容器为例来说明这个问题：
```rust
	let v = vec![1,2,3];
	println!("{}",v[0]);
```

vec!宏用于初始化一个vector容器。let叫做绑定，将值绑定在一个变量上。
v变量本身分配在栈里面，而v中的三个元素是分配在堆里面。当函数退出之后，v的作用域结束了，它所引用的堆中的元素也会被自动回收。听起来很不错，至少比malloc和free一定要配对强。

下面问题来了，如果想要将v的值绑定在另一个变量v2上，会出现什么情况呢？
对于有GC的系统来说，这不是问题，v和v2都引用同一个堆中的引用，最终由GC来回收就是了。
而对于没有GC的系统，有三种选择：
1. 复制：将堆中的数据重制一份，绑定到新的变量上。C++中的栈上的对象的赋值就是这么做的，会调用一个叫做拷贝构造函数的东西来负责这个复制。
2. 移动：C++11引入的std::move语义就是做这个的，将所有权转移。堆上的值被v2所引用，v的引用失效。
3. 引用计数：采用自动引用计数的方式，当引用计数为0的时候就释放掉堆上的数据。C+、+11和Boost的shared_ptr就是这种思想哈。

### Rust默认是移动语义

C++是默认以复制语义为主的，而Rust默认是移动语义。
在Rust中，如果引用了移动了控制权之后的对象中的元素，会是什么下场呢？答案是，根本编不过。
假如说我们这么写：
```rust
	let v = vec![1,2,3];
	let v2 = v;
	println!("{}",v[0]);
```

编译器就给我们报这样的错误，直接编译失败。
其实编译器给出的错误信息还是蛮友好的哈。

```
error[E0382]: use of moved value: `v`
 --> src/main.rs:7:16
  |
6 | 	let v2 = v;
  | 	    -- value moved here
7 | 	println!("{}",v[0]);
  | 	              ^ value used here after move
<std	macros>:2:27: 2:58 note: in this expansion of format_args!
<std	macros>:3:1: 3:54 note: in this expansion of print! (defined in <std macros>)
src/main.rs:7:2: 7:22 note: in this expansion of println! (defined in <std macros>)
  |
  = note: move occurs because `v` has type `std::vec::Vec<i32>`, which does not implement the `Copy` trait
```

编译失败告诉我们，除非实现了Copy trait，否则一个类型的默认是移动语义的。

好吧，你认了，默认移动就移动呗。小心啦，我们看看下面这个例子。
我们只是调个无害的函数嘛：
```rust
fn take(v: Vec<i32>){
	println!("I did nothing on v!");
}

fn main(){
	let v = vec![1,2,3];
	take(v);
	println!("{}",v[0]);
}
```

编译结果是什么？
```
error[E0382]: use of moved value: `v`
 --> src/main.rs:8:16
  |
7 | 	take(v);
  | 	     - value moved here
8 | 	println!("{}",v[0]);
  | 	              ^ value used here after move
<std	macros>:2:27: 2:58 note: in this expansion of format_args!
<std	macros>:3:1: 3:54 note: in this expansion of print! (defined in <std macros>)
src/main.rs:8:2: 8:22 note: in this expansion of println! (defined in <std macros>)
  |
  = note: move occurs because `v` has type `std::vec::Vec<i32>`, which does not implement the `Copy` trait
```

竟然跟上面的绑定到另一个变量上一样，v的所有权已经被别的变量给拿走了。

这可如何是好呢？
其实也没什么神秘的。正如带有GC的语言一般都会引用弱引用一样，这时需要的是一个不引起所有权转移的方式，就是Rust语言的引用。采用与C++类似的符号，在类型前加上&就表示引用啦。
改写成下面的引用的方式，就可以正常编译了：
```rust
fn take(v: &Vec<i32>){
	println!("I did nothing on v!");
}

fn main(){
	let v = vec![1,2,3];
	take(&v);
	println!("{}",v[0]);
}
```

在Rust里面，引用的方式有个新的名字叫做“borrow"-借用。只管用，不管生命周期的销毁。跟弱引用很类似，大家应该能很好的理解。

## Rust是不变性优先的语言

Rust语言在设计上，就是为线程安全多做考虑的一门语言。而什么样的数据在多线程下最安全呢？答案是不变的数据。Java世界的并发名著《Java Concurrency in Practice》在讲如何使用丰富的多任务工具之前，先讲了如何做状态封闭。
Rust的设计上也充分认识到了这一点。为什么其他语言是叫变量赋值，而在Rust就强调是变量绑定呢？其重要原因就是这个绑定默认是不变的。
比如我们默认定义个变量绑定，再对这个变量进行操作，会导致，编译不过。
我们试个最简单的例子：
```rust
    let a = 1;
    a = a + 2;
```

编译报下面的错：
```
  --> src\main.rs:10:5
   |
9  |     let a = 1;
   |         - first assignment to `a`
10 |     a = a + 2;
   |     ^^^^^^^^^ re-assignment of immutable variable
```

那么如果要定义可以改变的绑定可不可以呢？当然可以，但是在Rust里，这不是默认的行为，需要加mut关键字来说明，我们将上面的例子改写一下，在let后面加上mut：
```rust
    let mut a = 1;
    a = a + 2;
    println!("{}",a);
```

这就是Rust不同于很多其他语言的地方，比如C++和Java，它们是默认是可变的，不可变的要加const和final。而Rust反过来了，不变的是正常的，可变的要加mut。

比如我想要给vector v增加一个值：
```rust
    let v = vec![1,2,3];
    v.push(4);
```

会导致编译失败：

```
 --> src\main.rs:7:5
  |
6 |     let v = vec![1,2,3];
  |         - use `mut v` here to make mutable
7 |        v.push(4);
  |        ^ cannot borrow mutably
```

也是需要改成mut的才可以push:
```rust
    let mut v = vec![1,2,3];
    v.push(4);
```

有了默认是不变性的知识，我们再回头来看导致所有权转移的这条语句：
```rust
    let v = vec![1,2,3];
    let v2 = v;
```

实在是有点不人道啊，因为v和v2都不是mut的，都是常量啊。

为什么两个常量还是不可以共享同一份堆数据呢？答案很简单，因为两个不变量的生命周期可能是不一致的，默认的变量绑定可是没有引用计数这样的功能的，它是一直跟一个变量的生命周期所绑定的。

同样，引用默认也是不变的，要想改变引用的值，也需要加mut。
比如下面的还是编译不过的：
```rust
fn take(v: &Vec<i32>){
    v.push(4);
}
```

报同样的错：
```
error: cannot borrow immutable borrowed content `*v` as mutable
 --> src\main.rs:3:5
  |
3 |     v.push(4);
  |     ^
```

我们同样要用mut来修饰它，将凡是&出现的地方都换成&mut
```rust
fn take(v:  &mut Vec<i32>){
    v.push(4);
}

fn main(){
	let mut v = vec![1,2,3];
	take(&mut v);
	println!("{}",v[3]);
}
```

输出结果是4，终于大功告成了！

## 多重借用

下面我们继续思考一个问题，刚才都是只有一个借用者，只有一个引用，那么有多个引用会如何？
比如这样是不是可以？
```rust
	let mut v = vec![1,2,3];
	take(&mut v);
	println!("{}",v[3]);

    let v2 = &v;
    let v3 = &v;
```

是的，上面这些是可以的。

那么既有不变的，又有可变的，可不可以呢？
```rust
    let v2 = &v;
    let v3 = &v;
    let mut v4 = &mut v; 
```

结果是，又报错编不过了：
```
error[E0502]: cannot borrow `v` as mutable because it is also borrowed as immutable
  --> src\main.rs:12:23
   |
10 |     let v2 = &v;
   |               - immutable borrow occurs here
11 |     let v3 = &v;
12 |     let mut v4 = &mut v;
   |                       ^ mutable borrow occurs here
13 | }
   | - immutable borrow ends here
```

这又是Rust对于线程安全的考虑，如果大家都是只读的引用，可以有任意多个。但是只要有一个是可变的，对不起，那就只有这一个可以引用。违反了这条规则，不是运行时出问题，而是编译都不能通过。

那么一段代码中，业务逻辑就是要多于一种可变的引用来操作它怎么办?
解决方案是，分时操作。只要不是同时针对一个变量作引用就好了，我们可以通过作用域的方式将它们进行分时，比如这样做：
```rust
    {
        let v2 = &v;
        let v3 = &v;
    }
    {
        let mut v4 = &mut v;
    }
```
只加了两对括号，顺序上没有任何明影响。但是因为错开了作用域的重叠，就可以顺利编译通过，而且也避免了风险。

但是，要注意，一个长的作用域的引用，想要引用一个生命周期比它短的对象是编译不过的。
比如上面我们是在大作用域内定义对象，子作用域里定义引用，我们反过来，在大定义域里定义引用，引用子作用域里的对象，像这样:
```rust
    let vv: &mut Vec<i32>;
    {
        let vv2 = vec![4,5,6];
        vv = &mut vv2;
    }
```

看起来不错，但是编不过：

```
  --> src\main.rs:21:19
   |
21 |         vv = &mut vv2;
   |                   ^^^
   |
note: reference must be valid for the block suffix following statement 5 at 18:26...
  --> src\main.rs:18:27
   |
18 |     let vv: &mut Vec<i32>;
   |                           ^
note: ...but borrowed value is only valid for the block suffix following statement 0 at 20:30
  --> src\main.rs:20:31
   |
20 |         let vv2 = vec![4,5,6];
   |                               ^
```

## 小结一下

* Rust没有GC
* Rust的对象，默认是移动语义。在特殊情况下也支持复制语义。但是不管是移动还是复制，变量之间不能共享同一份堆上的数据
* 要想共享对象的数据，需要通过引用来借用。同一作用域内同时只能有一个可变的引用。没有可变的引用时，支持多个不变引用共享。
* Rust的变量绑定默认是不变的，可变的需要加mut关键字说明。引用也是如此。
* 较大作用域的引用不能引用活得比它短的小作用域内的对象，反之可以。
