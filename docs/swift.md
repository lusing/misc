# Swift笔记

常量和变量
let定义常量
var定义变量

数据类型

Int8
Int16
Int32
Int64
UInt8
UInt16
UInt32
UInt64

Float
Float32
Float64
Float80
Double

类型转换，使用类型的构造函数。

```swift
let a3 = UInt64(a1) + UInt64(a2);
```


Bool类型，true/false

元组

```swift
var a10:(Float32,String) = (1.0,"hhh");
var (num,str1) = a10;
```

Optional类型
在类型后面加?

使用!进行拆包

Character类型，9个字节

空合并运算符 ??

## 流程控制

for in循环

while循环

repeat while循环

if else

switch case

continue

break

fallthough

return

throw

guard

```swift
func fib(number n: Int64) -> Int64{
    guard (n>0) else{
        return 0
    }
    guard n != 1 && n != 2 else{
        return 1
    }
    return fib(number: n-Int64(1))+fib(number: n-Int64(2))
}
```

do...catch处理异常

try?将异常转化为Optional

defer

