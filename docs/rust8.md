# Rust语言教程(8) - 所有权

我们从第2讲到第7讲这6讲，在堆上分配了不少对象，但是类似于C++的delete运算符仍然没有出场过。因为Rust有作用域规则，在超出作用域之后会自动释放掉。

## 移动语义和复制语义

我们来复习下移动语义。

先看一段代码：
```rust
    let mut c_0 = Complex{real:0, imagine:0};
    println!("{}",c_0);
    println!("{}",c_0.imagine);
    println!("{:?}",c_0);
    c1.add(c_0);
    println!("{}",c_0);
```

最后一个println!是编译不过的，因为c1.add调用时，已经将c_0的所有权给拿走了，报错信息如下：
```
error[E0382]: borrow of moved value: `c_0`
   --> src/main.rs:459:19
    |
454 |     let mut c_0 = Complex{real:0, imagine:0};
    |         ------- move occurs because `c_0` has type `Complex`, which does not implement the `Copy` trait
...
458 |     c1.add(c_0);
    |            --- value moved here
459 |     println!("{}",c_0);
    |                   ^^^ value borrowed here after move
```

因为默认是移动语义，如果在println!之前把它赋值给另一个变量c_00，则所有权就会被c_00给拿走：
```rust
    let mut c_0 = Complex{real:0, imagine:0};
    let c_00 = c_0;
    println!("{}",c_0);
    println!("{}",c_0.imagine);
```

上面的代码编译不过，报错如下：
```
error[E0382]: borrow of moved value: `c_0`
   --> src/main.rs:457:19
    |
454 |     let mut c_0 = Complex{real:0, imagine:0};
    |         ------- move occurs because `c_0` has type `Complex`, which does not implement the `Copy` trait
455 |     let c_00 = c_0;
    |                --- value moved here
456 |     println!("{}",c_0);
457 |     println!("{}",c_0.imagine);
    |                   ^^^^^^^^^^^ value borrowed here after move
```

如果想要将当前的变量赋给新变量不影响使用，那么就可以克隆一个新对象出来。

支持克隆的话，我们需要支持Clone trait，跟Debug一样，让Rust帮我们生成：
```rust
    #[derive(Debug,Clone)]
    struct Complex {
        real : i32,
        imagine : i32
    }
```

然后Complex类就支持clone方法了：
```rust
    let c_00 = c_0.clone();
```

我们通过add方法把c_0的所有权转移走，可以看到clone出来的c_00的所有权不受影响：

```rust
    c1.add(c_0);
    println!("{}",c_00);
```

我们还可以更彻底一些，将移动语义变为复制语义，方法是实现Copy trait:
```rust
    #[derive(Debug,Clone,Copy)]
    struct Complex {
        real : i32,
        imagine : i32
    }
```

现在我们直接赋值，执行的也是复制语义了：

```rust
    let mut c_0 = Complex{real:0, imagine:0};
    let c_00 = c_0.clone();
    let c_01 = c_0;
```

就算是调用add方法，也仍然不会造成所有权转移了：
```rust
    c1.add(c_0);
    println!("{}",c_0);
```

当然，我们也可以不用Rust编译器的derive扩展，自己实现Clone trait和Copy trait. 

Clone trait只有一个方法需要被实现：
```rust
pub trait Clone {
    fn clone(&self) -> Self;
}
```

我们来实现一个：
```rust
    impl std::clone::Clone for Complex{
        fn clone(&self) -> Self {
            return Complex{real:self.real, imagine: self.imagine};
        }
    }
```

Copy trait更省事了，它是继承自Clone:
```rust
pub trait Copy: Clone { }
```

所以我们写个空的就好了：
```rust
    impl std::marker::Copy for Complex{   
    }
```

请大家注意std::marker这个包名，这里面的trait其实隐含了对于编译器的配置说明。std::marker::Copy并不只是一个空的继承自Clone的trait，还是对编译器语义转变的提示。

## 智能指针Box

除了自己管理所有权，我们还可以将所有权交给智能指针Box. 

我们来看一个用Box管理我们之前定义的Complex的例子：
```rust
    let c5 = Complex{real: 5, imagine: -1};
    let c51 = std::boxed::Box::new(c5);
```
c51的类型是Box<Complex>类型。

如果要访问c51的域，就像访问c5一样：
```rust
    println!("{}",c51.real);
```

如果要将c51当成Complex对象用的话，需要使用*运算符来访问：
```rust
    c1.sub(*c51);
    println!("{}",c1);
```

## 引用计数指Rc和Arc

另外一种管理共用所有权的方式是使用引用计数Rc和Arc，它们唯一的区别在于rc是线程不安全的，只能用于单线程情况；而arc是支持线程安全的，但是在单线程情况下没有必要为其花费额外的开销。

Rc的用法跟Box差不多：

```rust
    let c41 = std::rc::Rc::new(*c51);
    c1.sub(*c41);
    println!("{}",c1);
```

我们将其扔到一个线程里去试试效果，还记得写线程的方法吗？
```rust
    let child3 = thread::spawn(move || {
        println!("{}",c41);
    });
```

就会报下面这样的错：
```
error[E0277]: `Rc<Complex>` cannot be sent between threads safely
   --> src/main.rs:454:18
    |
454 |       let child3 = thread::spawn(move || {
    |  __________________^^^^^^^^^^^^^_-
    | |                  |
    | |                  `Rc<Complex>` cannot be sent between threads safely
455 | |         println!("{}",c41);
456 | |     });
```

要实现线程安全，我们可以使用std::sync::Arc来替换掉Rc:

```rust
    let ac1 = std::sync::Arc::new(c1);
    let child4 = thread::spawn(move || {
        println!("{}",ac1);
    });
```

## 小结

这一节如果要记住一个概念的话就记住Box吧，初学阶段默认可以使用它来控制所有权。
