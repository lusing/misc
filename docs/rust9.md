# Rust语言教程(9) - if let表达式与枚举进阶

## 枚举复习

前面我们介绍了Rust中的枚举类型，以及通过枚举实现的Option, Result等类型。
温故而知新，我们再来复习一下枚举的定义和使用。

首先，Rust的enum可以像C语言中的enum一样，定义一组可枚举的常量值。比如我们可以这样描述带符号的整数：

```rust
enum Integers{
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    Bigint
}
```

我们在使用时可以将常量转换为整数来使用，例如：

```rust
    let int1 = Integers::Int8;
    println!("{}",(int1 as i32));
```

如同C语言一样，Rust的enum常量也可以指定值：

```rust
enum Integers{
    Int8=1,
    Int16,
    Int32,
    Int64,
    Int128,
    Bigint
}
```

除了可以定义常量枚举，Rust还可以定义带有变量的枚举。例如，我们想描述一个8位的整数，它可以是有符号的，也可以是无符号的，我们可以这样定义：

```rust
enum Bits8 {
    Uint8(u8),
    Int8(i8)
}
```

我们就可以用这个Bits8枚举用来既能表示有符号的8位整数，也可以表示无符号的8位整数。为了共享，我们再加上前一节学的Box包装一下：

```rust
    let bit1 = Box::new(Bits8::Uint8(1u8));
    let bit2 = Box::new(Bits8::Int8(-2i8));
```

对于枚举的变量，我们可以通过match模式匹配的方法来对其进行计算。例如：

```rust
    match *bit1 {
        Bits8::Uint8(value) => println!("unsinged int:{}",value),
        Bits8::Int8(value) => println!("signed int:{}",value)
    }
```

当然，针对不匹配的情况，也可以什么也不错。这时候可以使用"_"来匹配不想处理的所有情况：

```rust 
    match *bit2 {
        Bits8::Int8(value) => println!("signed int:{}",value),
        _ => {}
    }
```

## if let表达式

因为match表达式要符合MECE原则，所以必须要处理所有的情况。为了代码写起来方便，像Rust和Swift这样的现代语言都实现了if let表达式。
以上面的例子为例，我们的if let表达式可以写为：

```rust
    if let Bits8::Uint8(value) = *bit1 {
        println!("unsinged int:{}",value);
    }
```

if let表达式当然也支持else：
    
```rust
        if let Bits8::Uint8(value) = *bit1 {
            println!("unsinged int:{}",value);
        } else {
            println!("not unsinged int");
        }
```

我们将bit1改成bit2，就会走else分支。

if let表达式最常用的情况就是在Option或者Result中使用。

比如我们将一个整数放到Option中，然后打包到Box里面：

```rust
let num1 = Box::new(Some(1i32));
```

如果要用到包装的值，暴力的作法是直接unwrap: 
```rust
let num2 = num1.unwrap() + 1i32;
```

但是这样的话，如果num1是None的话，就会panic。所以我们可以用if let来判断一下：

```rust
    if let Some(num3) = *num1 {
        let num4 = num3 + 2i32;
        println!("{} {} {}",num2,num3,num4);
    }
```

我们再看一个例子，将Option转化为Result:

```rust
fn test(i:Option<Box<i32>>) -> Result<Box<i32>, &'static str>{
    match i{
        Some(i) => Ok(i),
        None => Err("NAN")
    }
}
```

## 为枚举定义方法

以前面的Integers为例，我们不想将类型常量转换为整数来使用，而是想直接使用它们。这时我们可以为枚举定义方法，就像给struct定义方法一样，仍然使用impl关键字来实现。比如我们可以实现Display方法，来让它们可以直接打印：

```rust
impl Display for Integers{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", (*self as i32))
    }
}
```

针对带变量值的枚举类型，为其定义函数的作用将更大。

比如我们前面定义的统一的8位有符号和无符号整数的枚举值，我们现在就可以为其定义算数运算了。

```rust
impl Add for Bits8{
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        match (self, other) {
            (Bits8::Uint8(a), Bits8::Uint8(b)) => Bits8::Uint8(a + b),
            (Bits8::Int8(a), Bits8::Int8(b)) => Bits8::Int8(a + b),
            (Bits8::Uint8(a), Bits8::Int8(b)) => Bits8::Int8(a as i8 + b),
            (Bits8::Int8(a), Bits8::Uint8(b)) => Bits8::Int8(a + b as i8),
        }
    }
}
```

Add是对应“+”的trait，我们可以为Bits8实现Add，然后就可以直接使用“+”了。

```rust
    let bit4 = Box::new(Bits8::Uint8(1));
    let bit5 = Box::new(Bits8::Int8(-1));
    let bit6 = *bit4 + *bit5;
    println!("bit6:{}",bit6);
```

我们还可以使用上节学习的Arc将枚举值封装到多线程共享的变量中，这时在进行加法之前我们先将其borrow出来：

```rust
    let bit4 = Arc::new(Bits8::Uint8(1));
    let bit5 = Arc::new(Bits8::Int8(-1));
    let b4: &Bits8= bit4.borrow();
    let b5: &Bits8= bit5.borrow();
    let b6 = b4 + b5;
    println!("Arc:{:?}",b6);
```

borrow出来的结果是引用，没法使用上面针对于Bits8的Add方法，所以我们需要再定义一个Add方法，针对于引用。
返回值的时候我们就不要返回引用了，将其分配到Arc中再返回：

```rust
impl Add for &Bits8{
    type Output = Arc<Bits8>;
    fn add(self, other: Self) -> Self::Output {
        match (self, other) {
            (Bits8::Uint8(a), Bits8::Uint8(b)) => Arc::new(Bits8::Uint8(a + b)),
            (Bits8::Int8(a), Bits8::Int8(b)) => Arc::new(Bits8::Int8(a + b)),
            (Bits8::Uint8(a), Bits8::Int8(b)) => Arc::new(Bits8::Int8((*a) as i8 + b)),
            (Bits8::Int8(a), Bits8::Uint8(b)) => Arc::new(Bits8::Int8(a + (*b) as i8)),
        }
    }
}
```

## 小结

Rust的枚举是可以定义方法的，这样会大大有助于我们像使用struct一样使用枚举。
另外，if let也是非常有用的，可以让我们在不使用match的情况下，对枚举值进行判断和处理，可以大大简化代码，提升编程效率。
同样，也有while let这样的语法糖。
