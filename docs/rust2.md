# Rust语言教程(2) - 从熟悉的部分开始

虽然有默认不变性还有所有权的问题让Rust一上来用起来有些不同，但是其实大部分语法特点还是我们所熟悉的。
我们没必要上来就跟自己死磕，可以先从我们熟悉的部分开始学习。

一般我们写代码，使用的主要是数据类型、控制结构和函数。我们就从这三部分开始。

## 数据类型

与Go一样，Rust的定义语句数据也是放在变量名后面的，不过还要加上一个冒号。

### 布尔类型

布尔类型是bool：
```rs
let b0 : bool = true;
```

因为Rust是有类型推断的功能，所以很多时候可以不用指定类型。

```rust
    let b1 = true;
    let b2 = !b1;
    let b3 = 1 > 0;
    println!("{} {} {}",b1,b2,b3);
```

如果使用CLion等IDE的话，就可以直接看到IDE提供的灰色的类型推断的提示，非常方便：
![CLion.png](https://upload-images.jianshu.io/upload_images/1638145-636746ccf2909edf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


### 字符类型 - 传统与现代的结合

Rust的字符类型支持的是Unicode类型，占用4个字节。同时，Rust也支持单字节ASCII值，这时用b开头，类型值就是8位无符号类型u8。

我们来看例子：
```rs
    let c1 :char = 'C';
    let c2:u8 = b'C';
    let c3 = '中';
    println!("{} {} {}",c1,c2,c3);
```

同样，我们可以将字符组成字符串，我们来看例子：
```rs
    let s1 = "Hello";
    let s2 = b"World";
    println!("{} {:?}",s1,s2);
```
输出结果为：
```
Hello [87, 111, 114, 108, 100]
```

s1的真实类型是str类型，而s2是u8的数组。

```rs
    let s1 :&str = "Hello";
    let s2 :&[u8;5] = b"World";
```

### 整数类型: 后缀与下划线齐飞

按照长度，Rust的整数类型支持8位，16位，32位，64位，128位。根据有符号和无符号，分为有符号的i8,i16,i32,i64,i128和无符号的u8,u16,u32,u64,u128。
除此之外，也有根平台相关的类型，有符号为isize类型，无符号为usize类型。

我们看下例子：
```rs
    let i1 : i8 = -8;
    let i2 : i16 = -16;
    let i3 : i32 = -32;
    let i4 : i64 = -64;
    let i5 : i128 = -128;

    let u1 : u8 = 8;
    let u2 : u16 = 16;
    let u3 : u32 = 32;
    let u4 : u64 = 64;
    let u5 : u128 = 128;
    
    let p1 : isize = -1;
    let p2 : usize = 1;
```

上面都是跟其它语言比较像，下面我们来看看Rust特色的后缀。这在C++中也有，比如10l, 200L之类的。
在Rust中，我们直接用类型名做为后缀，我们看个例子：
```rs
    let i6 = -1i8;
    let i7 = -2i16;
```

这样放在一起可能不太容易区分，没关系，Rust允许我们在数字上任意的加下划线来提升可读性，我们来看几个例子：
```rs
    let i08 = -3_i32;
    let i09 = -4__i64;
    let i10 = -5___i128;
```

下划线并非只是用于数字和类型区分，也可以加在数字中间，我们来看个例子：
```rs
    let u6 = 1_000_000_u128;
    println!("{}",u6);
```

默认的整数类型是i32，如果Rust无法推断中整数的类型，那么就默认为i32.

### 整数的进制

在Rust中，避免了077这样对八进制的偏爱，改为用0o来表示8进制整数。16进制仍然是0xFF前缀，二进制用0b前缀。

我们看例子：
```rs
    let u07 = 0xFF_u32;
    let u08 = 0o7777_u32;
    let u09 = 0b01_10_00_u8;
    println!("{} {} {}",u07,u08,u09);
```

输出结果为：
```
255 4095 24
```

### 整数的溢出

在C语言中，整数的溢出也是一个常出现的问题。
对此，Rust在debug模式下，在编译时会检查整数的溢出的问题：
```rs
    let i_10 : i8 = 0x7f;
    let i_11 : i8 = i_10 * 10i8;
    println!("{}",i_11);
```

在编译时，Rust就会报错：
```rs
84 |     let i_11 : i8 = i_10 * 10i8;
   |                     ^^^^^^^^^^^ attempt to compute `i8::MAX * 10_i8`, which would overflow
```

懂程序分析的同学可能会想，在编译时检查不出来怎么办？好办，我们在运行时进行检查。
我们来个例子：
```rs
    let mut i_20 : i8 = 0x20;
    for i in 1..20{
        i_20 = 0x20_i8 * i_20;
    }
    println!("{}",i_20);
```

在运行时仍然发现了溢出：

```
thread 'main' panicked at 'attempt to multiply with overflow', src/main.rs:91:16
stack backtrace:
   0: rust_begin_unwind
             at /rustc/7eac88abb2e57e752f3302f02be5f3ce3d7adfb4/library/std/src/panicking.rs:483
   1: core::panicking::panic_fmt
             at /rustc/7eac88abb2e57e752f3302f02be5f3ce3d7adfb4/library/core/src/panicking.rs:85
   2: core::panicking::panic
             at /rustc/7eac88abb2e57e752f3302f02be5f3ce3d7adfb4/library/core/src/panicking.rs:50
   3: tools::test
             at ./src/main.rs:91
   4: tools::main
             at ./src/main.rs:34
   5: core::ops::function::FnOnce::call_once
             at /Users/lusinga/.rustup/toolchains/stable-x86_64-apple-darwin/lib/rustlib/src/rust/library/core/src/ops/function.rs:227
note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace.
```

### 类型转换

Rust语言是强类型的语言，不像C一样有默认的类型转换。如果进行跨类型计算需要进行类型转换。
类型转换使用“as 类型”的方法来写，我们来看个例子：
```rs
let i_100 : i32 = (16i8 + 1) as i32;
```

i8计算之后还是i8，不能直接赋给i32类型，需要通过as i32来转换类型。

如果计算的类型不同，编译不报错，在运行的时候也会被检查出来。
我们看个例子：
```rs
let i_101 = 16i8 + 32i32;
```

会报下面的错：
```
error[E0308]: mismatched types
  --> src/main.rs:99:24
   |
99 |     let i_101 = 16i8 + 32i32;
   |                        ^^^^^ expected `i8`, found `i32`
```
后面还有一个有趣的报错，让trait露了一个爪印：
```
error[E0277]: cannot add `i32` to `i8`
  --> src/main.rs:99:22
   |
99 |     let i_101 = 16i8 + 32i32;
   |                      ^ no implementation for `i8 + i32`
   |
   = help: the trait `Add<i32>` is not implemented for `i8`
```

### 浮点数

浮点数跟C语言差不多，分为32位浮点数和64位浮点数，就这两种，分别是f32和f64。默认为f64。
我们来看两个例子：
```rs
    let f_01 = 2.1;
    let f_02 = 2e8;
```

f_01和f_02都是f64类型。

需要注意的是，对于除0的处理，会引入两个新的值：
- 对于非0除以0，得到的将是无穷大inf
- 而对于0除以0，将得到NaN，意思是并不是一个数

我们来看例子：
```rs
    let f_03 = 0.0 / 0.0;
    let f_04 = 1.0 / 0.0;
    println!("{} {}",f_03,f_04);
```
输出结果为：
```
NaN inf
```

NaN对应的本尊是std::f64::NAN，而inf是std::f64::INFINITY，我们将其排列在一起：

```rs
    let f_03 = 0.0 / 0.0;
    let f_04 = 1.0 / 0.0;
    let f_05 = std::f64::INFINITY;
    let f_06 = std::f64::NAN;
    println!("{} {} {} {}",f_03,f_04,f_05,f_06);
```

输出结果为：
```
NaN inf inf NaN
```

32位和64位的无穷大都是无穷大，它们是相等的：
```rs
    let f_10 = std::f32::INFINITY;
    let f_11 = std::f64::INFINITY;
    println!("{}",f_11==f_10 as f64);
```

输出结果为：
```
true
```

但是要注意的是，两个NAN是不相等的：
```rs
    let f_12 = std::f64::NAN;
    println!("{}",f_12==f_12);
```

结果为false.

## 流程控制

### 分支语句

Rust支持if-else表达式，用来处理分支。
if后面不必加括号，有点像Go，我们看个例子：
```rs
    if n >= 100 {
        println!("Grade A");
    }else if n>= 60 {
        println!("Pass");
    }else{
        println!("Fail");
    }
```

可以写成更像表达式一点的方式：
```rs
    let grade = if n == 100{
        "A"
    }else if n>=60{
        "Pass"
    }else{
        "Fail"
    };
```

如果用作表达式的话，if和else两个分支返回的结果需要转换成同一类型，毕竟Rust是这么强类型的语言。

### 循环语句

Rust的循环分为三种：死循环loop，while循环和for循环。

loop最直接干脆，不需要`while(true)`或者`for(;;)`这种写法，直接loop。如果需要退出循环就用break，继续下一轮循环就用continue。

我们来个简单例子：
```rs

    let mut num = 0;
    let mut sum = 0;
    loop{
        if num > 10 {
            break;
        }else{
            sum += num;
            num += 1;
        }
    }
    println!("sum={}",sum);
```

我们再将其翻译成while循环：
```rs
    num = 0;
    sum = 0;
    while num <= 10 {
        sum += num;
        num += 1;
    }
    println!("sum={}", sum);
```

与if一样，while后面也不强制要求括号。

最后是for循环，它主要用于迭代器的遍历：
```rs
    sum = 0;
    for i in 0..11  {
        sum += i;
    }
    println!("sum={}", sum);
```

## 函数

最后说下函数，Rust的函数使用fn关键字来定义。返回值的类型用->分隔而不是":"。
另外，Rust中不一定非要用return语句来返回值，表达式的值即可，我们看个例子：
```rs
fn fib2(n: i32) -> i64 {
    if n <= 2 {
        1i64
    } else {
        fib2(n - 1) + fib2(n - 2)
    }
}
```

按传统写法也是可以的：
```rs
fn fib2(n: i32) -> i64 {
    if n <= 2 {
        return 1i64
    } else {
        return fib2(n - 1) + fib2(n - 2)
    }
}
```

或者将return提到if表达式外面：
```rs
fn fib2(n: i32) -> i64 {
    return if n <= 2 {
        1i64
    } else {
        fib2(n - 1) + fib2(n - 2)
    }
}
```

## 小结

在使用基本类型的情况下，Rust跟C语言和Go语言的基础部分其实还是很类似的，熟悉Javascript等语言的同学也不会觉得陌生。我们可以把原有的知识迁移过来，基本类型变量如果需要修改值的话就加个mut。

