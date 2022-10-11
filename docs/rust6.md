# Rust语言教程(6) - 错误处理和可选值

## Rust的错误处理

从前面的学习中，我们对于Rust的错误处理应该已经有个体感了。Rust的返回值中，同时包含了正确情况下的值和错误情况下的报错信息的值。

以上一讲的读取标准输入值为例，我们其实没有处理返回值：
```rust
    let sin1 = std::io::stdin().read_line(&mut str8);
```

如果要处理的话，需要如何做呢？

### 使用match表达式处理错误

我们可以使用match表达式，来同时对正确和错误两种情况进行处理。
我们来看处理上面的返回值的例子：
```rust
    println!("Please input a number:");
    let mut str8 = String::with_capacity(255);
    let sin1 = std::io::stdin().read_line(&mut str8);
    match sin1 {
        Ok(size1) =>{
            println!("The size read from stdin is:{}",size1);
            let num8 = str8.trim().parse::<i32>().unwrap();
            println!("{}",num8);
        }
        Err( err) =>{
            println!("Read error!{}",err);
        }
    }
```

### 通过判断状态处理错误

如果觉得match表达式嵌套得太多了看起来烦，没关系，我们也可以直接通过判断状态来进行处理。状态不看就unwrap有点太武断了，早晚panic，加个is_ok的判断吧：

```rust
            let rnum = str8.trim().parse::<i32>();
            if rnum.is_ok(){
                println!("{}",rnum.unwrap());
            }
```

在这种情况下如何获取错误信息呢？我们可以通过err方法。但是，err方法返回的是一个Option可选值，与错误值类似，我们还需要加一重处理，我们来看例子：
```rust
            let rnum = str8.trim().parse::<i32>();
            if rnum.is_ok(){
                println!("{}",rnum.unwrap());
            }else{
                let err1 = rnum.err();
                if err1.is_some(){
                    println!("{}",rnum.err().unwrap());
                }
            }
```

### 错误时给个默认值

有的时候，有错误也不重要，我们给一个默认值就好。

比如上面的parse整数，如果失败就给个0值，提升容错性：
```rust
println!("{}",rnum.unwrap_or(0));
```

当然，如果只给个默认值，可能还不够，我们还想做一些进一步的操作。这时，我们可以给一个闭包，在闭包里实现更多的控制，比如可以打印个日志之类的：
```rust
            rnum.unwrap_or_else(|err| {
                return 0;
            });
```

### panic时给出定制化的提示

像上面的输入的例子，如果用户给出的值不合法，将程序panic掉也是一种办法。
但是我们可以输出自己的提示，这样用户知道是什么问题，下一次输入的时候可以避免。

我们还是修改上面的例子：
```rust
            println!("The size read from stdin is:{}",size1);
            let rnum = str8.trim().parse::<i32>();
            rnum.expect("Please input an integer");
```

输出的panic信息如下：
```
thread 'main' panicked at 'Please input an integer: ParseIntError { kind: InvalidDigit }', src/main.rs:303:18
stack backtrace:
   0: rust_begin_unwind
             at /rustc/e1884a8e3c3e813aada8254edfa120e85bf5ffca/library/std/src/panicking.rs:495:5
   1: core::panicking::panic_fmt
             at /rustc/e1884a8e3c3e813aada8254edfa120e85bf5ffca/library/core/src/panicking.rs:92:14
   2: core::option::expect_none_failed
             at /rustc/e1884a8e3c3e813aada8254edfa120e85bf5ffca/library/core/src/option.rs:1268:5
   3: core::result::Result<T,E>::expect
             at /Users/lusinga/.rustup/toolchains/stable-x86_64-apple-darwin/lib/rustlib/src/rust/library/core/src/result.rs:933:23
   4: tools::test
             at ./src/main.rs:303:13
   5: tools::main
             at ./src/main.rs:36:5
   6: core::ops::function::FnOnce::call_once
             at /Users/lusinga/.rustup/toolchains/stable-x86_64-apple-darwin/lib/rustlib/src/rust/library/core/src/ops/function.rs:227:5
note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace.
```

## 复合类型

随着学习的深入，靠简单类型可能已经难以满足我们的需求了。我们现在可以引入一些复合类型来方便编程。

### 元组

同其它语言一样，Rust的元组tuple也是用括号括起来的一组可以为不同类型的值的组合。

我们来个例子：
```rust
    let t1 = (1.0, "Test");
```

要访问元素的话，可以使用数字做下标，比如第1个就是t1.0：
```rust
    let t1 = (1.0, "Test");
    println!("{} {}",t1.0,t1.1);
```

元组可以为空，()叫做unit。

### 结构体

元组只有个顺序，如果想给字段指定个名字的话，可以使用结构体。

比如我们定义实部和虚部构成的复数：
```rust
    struct Complex {
        real : i32,
        imagine : i32
    }
```

初始化可以使用下面的方法：
```rust
let c1 = Complex{real :0, imagine: 1};
```

访问的话大家都很熟悉了：
```rust
println!("{}+{}i",c1.real,c1.imagine);
```

### 枚举enum

Rust的enum有点意思，用来表示可以是这种类型，也可以是另一种类型的类型。
比如我们可以定义一种既可能是数字又可能是字符串的类型：
```rust
    enum NumberOrText {
        Int(i32),
        Text(String)
    }
```

其中的字段名可以用来初始化类型变量的值：
```rust
    enum NumberOrText {
        Int(i32),
        Text(String)
    }

    let e1 = NumberOrText::Int(1);
    let e2 = NumberOrText::Text("1".to_string());
```

那么，这个有两种可能的类型如何处理呢？我们还是用match表达式来处理：
```rust
    match e1 {
        NumberOrText::Int(value) => println!("{}",value),
        NumberOrText::Text(value) => println!("{}",value)
    }
```

## 可选值

在Rust中，如果返回值可能为空，可以使用可选值来封装之。这样就可以避免很多其它语言可能出现的NullPointerException之类的错误。
前面介绍的enum虽然看起来有点奇怪，但是却是实现可选值的有力武器。
没错，Rust的Option类型就是用enum来实现的：
```rust
pub enum Option<T> {
    None,
    Some(T),
}
```

我们来看个例子，可能是整数，也可能是空值的类型，可以用Option<i32>等表示：
```rust
    let o1 : Option<i32> = Some(1_i32);
    let o2 : Option<i32> = None;
```

如何使用Option呢？我们可以使用is_some来判断是不是有值，有的话就可以unwrap：
```rust
    let o3 = Some(2_i32);
    if o3.is_some(){
        println!("{}",o3.unwrap());
    }
```

当然，不嫌烦的话，match表达式仍然是很好的选择：
```rust
    let o1 : Option<i32> = Some(1_i32);
    let v1 = match o1 {
        Option::Some(value) => value,
        Option::None => 0_i32
    };
```

同样，我们也可以用unwrap_or来指定默认值：
```rust
    let o2 : Option<i32> = None;
    println!("o2={}",o2.unwrap_or(0));
```

## 小结

有了enum的基础，我们再回望一下错误处理的Result类型，它其实也是个enum：
```rust
pub enum Result<T, E> {
    Ok(T),
    Err(E),
}
```
