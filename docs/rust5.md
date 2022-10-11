# Rust语言教程(5) - 与环境交互

学习了基本编程结构之后，我们还需要了解如何和外界环境打交道，这样才有能力做些有用的工具。

## 调用外部应用程序

通过标准库中的std::process::Command可以调用外部的应用程序或者命令。

我们来看个例子：
```rust
    let output = Command::new("ls")
        .arg("/Users/lusinga/working/llvm-project-11.0.0")
        .output();
    println!("{:?}",output.unwrap());
```

输出：
```
Output { status: ExitStatus(ExitStatus(0)), stdout: "CONTRIBUTING.md\nREADME.md\nclang\nclang-tools-extra\ncompiler-rt\ndebuginfo-tests\nflang\nlibc\nlibclc\nlibcxx\nlibcxxabi\nlibunwind\nlld\nlldb\nllvm\nmlir\nopenmp\nparallel-libs\npolly\npstl\nutils\n", stderr: "" }
```

返回的标准输出值在output.stdout里，它是一个u8的Vec，可以拿来进行处理。

## 创建新线程

过不了多久，生活在原生多核时代的同学们就将走上工作岗位和我们一起编程了，所以这时候还有单任务确实说不过去了。
Rust的多任务支持说来话长，而且也还在迭代改进中，不过基本上大家熟悉的thread, future, async这些都有，用起来也不复杂。

首先最简单的还是创建一个新线程。 创建新线程使用thread::spawn方法，参数是一个闭包。这个闭包不需要参数。
我们来看个最简单的例子：
```rust
    let child1 = thread::spawn(|| {
        println!("Hello,from thread");
    });
```

调用线程之后我们可以给个返回值，比如我们改成这样：
```rust
    let child1 = thread::spawn(|| {
        println!("Hello, from new thread!");
        return "From thread";
    });
```

返回类型是一个JoinHandle<&str>类型，我们可以通过JoinHandle的join方法去读取：
```rust
    let str1 = child1.join().unwrap();
    println!("In main thread - {:?}",str1);
```

输出结果是这样的：
```
Hello, from new thread!
In main thread - "From thread"
```

最后，如果想引用外面的变量怎么办？这时候我们需要获取所有权了，所以我们需要将闭包改成move闭包，就是在闭包前面加上move来修饰：

```rust
fn test_thread(){
    let str1 = "Hello, defined in main thread";
    let child1 = thread::spawn(move || {
        println!("{}",str1);
        return "From thread";
    });

    let str1 = child1.join().unwrap();
    println!("In main thread - {:?}",str1);
}
```

## 读取目录

跟进程打完交道之后，我们开始跟文件系统进行交互。这会使用到std库中fs模块的一些类型。

首先我们使用Path来描述一个路径：
```rust
    let path1 = Path::new("/Users/lusinga/working/llvm-project-11.0.0");
    println!("{:?}",path1.file_name().unwrap());
```

输出为：
```
"llvm-project-11.0.0"
```

然后我们可以用read_dir方法去读取目录下的内容：
```rust
    let dir_result = path1.read_dir().unwrap();

    for entry_result in  dir_result{
        let entry  = entry_result.unwrap();
        println!("{:?}",entry.path());
    }
```

read_dir返回的是ReadDir类型，它是一个Result<DirEntry>的迭代器。我们使用for循环来遍历它，每个元素就是一个DirEntry。

## 读取标准输入

当我们学习C++的时候，在一开始就会学到cout和cin，做oj题目的时候，如果没有办法从标准输入读入，题都没法做。

我们可以通过std::io::stdin()来获取标准输入流，然后通过read_line方法来获取一行输入。

我们来看个例子：
```rust
    let mut str8 = String::with_capacity(255);
    let sin = std::io::stdin().read_line(&mut str8);
    println!("{}",str8);
```

read_line方法需要一个mut String的引用作为参数。

获取字符串之后，可以使用之前介绍的parse方法来转换。

我们来看个读取数字的例子：
```rust
    println!("Please input a number:");
    let mut str8 = String::with_capacity(255);
    let sin = std::io::stdin().read_line(&mut str8);
    let num8 = str8.trim().parse::<i32>().unwrap();
    println!("{}",num8);
```

## 小结

到目前为止，我们发现，即使对于Rust的很多核心不了解的情况下，我们仍然可以开始写很多代码了。毕竟Rust是工具而不是信仰，能够让更多同学将其用起来解决问题是第一要务。

