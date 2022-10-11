# Rust语言教程(4)  - 字符串

有了数组和向量的基础，我们再来看它的一个特例：字符串。

字符串有两种表现形式，一种是基本类型，表示字符串的切片，以&str表示；另一种是可变的string类型。

针对字面值的字符串，有一种静态的类型表示方法，写作&'static str。

字符串切片具有普通切片的基本功能。

## 字符串切片长度

与普通切片一样，可以用len方法来求字符串切片的长度。

例：
```rust
    let s1 = "Hello,String";
    println!("{:?}",s1.len());
```

输出结果为12. 

## 判断字符串切片是否为空

可以通过is_empty方法判断一个字符串切片是否为空。

```rust
    let s2 = "";
    println!("{}",s2.is_empty());
```

## 将字符串按行拆分

可以通过lines方法将字符串切片拆分成行的迭代，这样就可以用for循环去处理每一行。

我们看个小例子：
```rust
    let str3 = "Hello\nWorld";
    for str_1 in str3.lines(){
        println!("str_1={}",str_1);
    }
```

## 判断当前字符串是否包含某子串

contains函数用于判断某一个子串是否是当前字符串切片的子串。

我们来看个例子，看看""Hello,String"中是否有"ello"：
```rust
    let s1 = "Hello,String";
    println!("{}",s1.contains("ello"));
```
输出结果当然为true. 

特别的，如果判断是否是以某串起始，可以使用starts_with函数；同样，判断是否以某串结尾，可以使用ends_with函数。

例：
```rust
    println!("{}",s1.starts_with("/"));
    println!("{}",s1.ends_with(";"));
```

如果我们希望查找到第一次出现的位置，我们可以使用find方法。

我们看个例子：
```rust
    let s1 = "Hello,String";
    println!("{:?}",s1.find("l"));
```

只能给静态字符串还是太弱，针对复杂条件，我们还可以给一个函数进去，比如char包中给我们提供的判断字符串类型的方法都可以用上。比如查找第一个字符，我们不写正则表达式了，直接上char::is_whitespace函数：
```rust
println!("{:?}",s1.find(char::is_whitespace));
```

或者条件比较复杂，我们自己写个函数：
```rust
    fn is_great_o (x: char) -> bool {
        (x > 'O' && x <= 'Z') || (x > 'o' && x <= 'z')
    }
    println!("{:?}",s1.find(is_great_o));
```

懒得写函数了，直接在find里面写也没问题：
```rust
s1.find(|x| (x > 'O' && x <= 'Z') || (x > 'o' && x <= 'z'));
```

代码里的`|x| (x > 'O' && x <= 'Z') || (x > 'o' && x <= 'z')`叫做闭包，也就是匿名函数。“｜变量名｜”用以捕获外界的变量，后面是返回值。可能通过"->"指定类型，也可以让系统自己推断。

加上类型之后的写法是下面这样：
```rust
s1.find(|x :char | -> bool  {(x > 'O' && x <= 'Z') || (x > 'o' && x <= 'z')});
```

加上类型之后，为了区分函数体，我们需要给函数体加上花括号。

更进一步，我们不光想知道第一次出现的位置，需要所有的匹配的位置，可以使用matches方法。
我们来看例子，假如我们想得到所有的小写字母：
```rust
    for str in s1.matches(char::is_lowercase){
        println!("str={:?}",str);
    }
```

输出结果为
```
str="e"
str="l"
str="l"
str="o"
str="t"
str="r"
str="i"
str="n"
str="g"
```

更进一步，如果我们不光想遍历符合条件的字符串，还想同时知道符合条件的子串的位置，我们可以使用match_indices方法：
```rust
    for str in s1.match_indices(char::is_lowercase){
        println!("match={:?}",str);
    }
```

输出结果为:
```
match=(1, "e")
match=(2, "l")
match=(3, "l")
match=(4, "o")
match=(7, "t")
match=(8, "r")
match=(9, "i")
match=(10, "n")
match=(11, "g")
```

## 拆分成子字符串

这也是字符串中最常用的基本操作之一，根据某分隔符将子符串拆分成一些子串，可以使用split方法来实现。
我们看个例子：
```rust
    let str4 = "/workspace/xulun/alios";
    for str_2 in str4.split("/"){
        println!("str4={}",str_2);
    }
```

输出结果为：
```
str4=
str4=workspace
str4=xulun
str4=alios
```

## 截掉无用字符

我们从文件中或者是网络上获得的字符串，经常是前后带有空白符的。此时我们可以调用trim方法将前后的空白去掉。

```rust
    let str5 = "\t\t\tHello\n\r";
    println!("{:}",str5.trim());
```

如果只想trim前缀的空白，可以使用trim_start方法；同理，后缀的可以使用trim_end。

```rust
    let str5 = "\t\t\tHello\n\r";
    println!("{:}",str5.trim());
    println!("{:}",str5.trim_start());
    println!("{:}",str5.trim_end());
```

如果想要去除的不是空白符，而是想自定义的其他字符，Rust提供了trim_matches，跟matches一样，可以给字符，字符串切片，函数或者闭包。
同样，如果只针对前缀，可以使用trim_start_matches；后缀使用trim_end_matches.

例如，我们想把前缀的路径符号去掉：
```rust
    let str4 = "/workspace/xulun/alios";
    for str_2 in str4.split("/"){
        println!("str4={}",str_2.trim_start_matches('/'));
    }
```

## 字符串替换

刚才的trim系只能处理前缀和后缀，如果要处理整个字符串，那就是字符串替换该做的事情了。

我们来个例子，把字符串里的“l”全部替换成空：
```rust
println!("{:}",str5.replace("l",""));
```

## 生成重复的字符串

这个没啥技术含量了。直接上例子：
```rust
println!("{}","a".repeat(10));
```
输出：
```
aaaaaaaaaa
```

## 将字符串解析成其它类型

字符串的常用操作研究差不多了，我们研究下将字符串转成其它类型，最常用的是转换成数字。这要用到字符串切换的parse方法。

parse可能成功，也可能失败，所以返回的结果不是一个数，而是一个Result类型的结构。如果成功的话，返回Ok(数字)；如果失败的话，返回Err(错误类型). 

我们来看几个例子，比如我们想把字符串"4"解析成i32整数：
```rust
println!("{:?}","4".parse::<i32>());
```

输出结果为：
```
Ok(4)
```

如果我们想获取4这个值，可以调用unwrap方法：
```rust
println!("{:?}","4".parse::<i32>().unwrap());
```

下面我们再看如何处理错误的情况，比如我们拿到的字符串不是"4"而是"4.0"，解析出发的结果如下：
```rust
println!("{:?}","4.1".parse::<i32>());
```
输出如下：
```
Err(ParseIntError { kind: InvalidDigit })
```
可以看到，错误是InvalidDigit类的ParseIntError。关于Rust的错误处理，我们将在后面介绍。

## 可变的字符串String

这一节的最后，我们再简单介绍下可变的字符串。前面的字符串切片有点像Java中的String，而可变字符串就像是StringBuffer或者StringBuilder。
在Rust中，String类型其实就是对Vec<u8>的封装，本质上就是一个向量。

因为就是向量，所以我们可以用向量的with_capacity来创建一个字符串：
```rust
let mut str5 = String::with_capacity(10);
```

如果在String末尾增加一个字符，则就像向量添加元素一样使用push。如果要增加一个字符串切片的话，可以使用push_str方法：

```rust
    let mut str5 = String::with_capacity(10);
    str5.push_str("Hello");
    println!("{:?} {}",str5,str5.capacity());
```

比起从头构建String，从一个字符串切片生成字符串是更常见的做法，这就是常用的to_string方法:
```
let mut str6 = "Hello".to_string();
```

如果不是在末尾push_str，而是需要在中间插入字符串的话，可以使用insert_str方法：

```rust
    let mut str6 = "Hello".to_string();
    str6.insert_str(5,"World!");
    println!("{:?} {}",str6,str6.capacity());
```

## 小结

这一节我们集中介绍了Rust中的字符串切片&str和可变字符串String的用法。除了要修改的赋值要加mut之外，跟其它语言其实还是蛮相似的吧，隐隐约约还有点函数式编程的味道。
