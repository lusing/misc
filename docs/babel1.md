# ES6以上版本代码要不要转码成ES5?

为了兼容老的浏览器，尤其是IE系列，使用ES6以上规范的前端代码往往使用Babel等转码工具转码成ES5的代码。

距离发布ES6的2015年已经过去了6年了，现在浏览器对于ES6的兼容性如何呢？

我们来看下CanIUse的数据：
![](https://gw.alicdn.com/imgextra/i4/O1CN01RNHB1x27RFGtA010w_!!6000000007793-2-tps-2746-1502.png)

可以看到，有98.14%的浏览器支持ES6. 没有超过99%的原因是因为2015年发布的Opera Mini还有1.08%的使用率。
针对手机端，2016年以后发布的Safari on iOS和Chrome等全部都支持ES6.
Safari on iOS 7-9.3目前的用户占比0.15%. 
Android从5版本开始WebView已经全线支持ES6. 

从数据上看起来，因为数量很少的老设备导致近99%以上的设备能力没有应用起来，似乎并没有说服力。
另外，很多应用针对低端机有特殊的处理，中高端机一定是近期的老设备。至少针对中高端机，转码的兼容性必要性基本上是可以忽略的。

不过，ES6及以上版本是由多个功能组成的，不能简单抽象成6比5好。
我们将主要的功能转码和不转码做一个对比。

## 不转码效果更好

### const

const是要带常量检查的。我们来个例子：

```js
let f1 = () => {
  const a = 0;
  a = 2;
};
f1();
```

转码之后，Babel帮我们生成了一个_readOnlyError函数。

```js
function _readOnlyError(name) { throw new TypeError("\"" + name + "\" is read-only"); }

var f1 = function f1() {
  var a = 0;
  2, _readOnlyError("a");
};
f1();
```

这个不用看字节码了，看源码就知道是哪个更好了。

### 数组拷贝

ES6之后，我们做数组拷贝使用扩展运算符"...". 

```js 
  const a1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  let a2 = [...a1];
```

Babel做的转码也不含糊，一个concat函数搞定：
```js 
  var a1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  var a2 = [].concat(a1);
```

但是，从字节码角度来看，就不一样了。因为v8提供了CreateArrayFromIterable指令。
所以，转码之前，9个字节的指令就搞定了：
``` 
Bytecode length: 9
Parameter count 1
Register count 2
Frame size 16
OSR nesting level: 0
Bytecode Age: 0
         0x3c140829374e @    0 : 79 00 00 25       CreateArrayLiteral [0], [0], #37
         0x3c1408293752 @    4 : c4                Star0 
         0x3c1408293753 @    5 : 7a                CreateArrayFromIterable 
         0x3c1408293754 @    6 : c3                Star1 
         0x3c1408293755 @    7 : 0e                LdaUndefined 
         0x3c1408293756 @    8 : a9                Return 
Constant pool (size = 1)
0x3c1408293721: [FixedArray] in OldSpace
 - map: 0x3c1408002205 <Map>
 - length: 1
           0: 0x3c1408293715 <ArrayBoilerplateDescription PACKED_SMI_ELEMENTS, 0x3c14082936e5 <FixedArray[10]>>
```

转码之后就是函数调用了，还有生成一个空数组，一共需要21个字节：
``` 
         0x3c1408293696 @    0 : 79 00 00 25       CreateArrayLiteral [0], [0], #37
         0x3c140829369a @    4 : c4                Star0 
         0x3c140829369b @    5 : 7b 01             CreateEmptyArrayLiteral [1]
         0x3c140829369d @    7 : c1                Star3 
         0x3c140829369e @    8 : 2d f7 01 02       LdaNamedProperty r3, [1], [2]
         0x3c14082936a2 @   12 : c2                Star2 
         0x3c14082936a3 @   13 : 5e f8 f7 fa 04    CallProperty1 r2, r3, r0, [4]
         0x3c14082936a8 @   18 : c3                Star1 
         0x3c14082936a9 @   19 : 0e                LdaUndefined 
         0x3c14082936aa @   20 : a9                Return 
Constant pool (size = 2)
0x3c1408293665: [FixedArray] in OldSpace
 - map: 0x3c1408002205 <Map>
 - length: 2
           0: 0x3c1408293659 <ArrayBoilerplateDescription PACKED_SMI_ELEMENTS, 0x3c1408293629 <FixedArray[10]>>
           1: 0x3c1408202e9d <String[6]: #concat>
```

### String.raw

对于String.raw，转码也是会多生成函数的。
比如转码前是这样：
```js 
let f1 = () => {
  String.raw`\n`;
};

f1();
```

转码之后，Babel帮我们生成了一个_taggedTemplateLiteral函数：

```js
var _templateObject;

function _taggedTemplateLiteral(strings, raw) { if (!raw) { raw = strings.slice(0); } return Object.freeze(Object.defineProperties(strings, { raw: { value: Object.freeze(raw) } })); }

var f1 = function f1() {
    String.raw(_templateObject || (_templateObject = _taggedTemplateLiteral(["\n"])));
};

f1();
```

### Symbol

Symbol是ES6新增的数据类型。
在ES6里，我们要判断其类型，直接使用typeof运算符就好。

```js 
let f2 = () => {
  let s1 = Symbol();
  return typeof s1;
};
```

这可难为Babel了，都得引入一个库才能解决：
```js 
function _typeof(obj) { "@babel/helpers - typeof"; return _typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) { return typeof obj; } : function (obj) { return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }, _typeof(obj); }

var f1 = function f1() {
  var s1 = Symbol();
  return _typeof(s1);
};
```

### rest参数

为了支持rest参数，v8提供了CreateRestParameter指令。不过，原有的arguments也是有CreateMappedArguments指令支持的。
二者算是打平。

不过，从源代码的角度来看，不转码的会短一些：

```js 
let f1 = (...values) => {
    let sum = 0;
    for (let v of values) {
        sum += v;
    }
    return sum;
};
f1(1, 4, 9);
```

转码之后如下：
```js 
var f1 = function f1() {
    var sum = 0;

    for (var _len = arguments.length, values = new Array(_len), _key = 0; _key < _len; _key++) {
        values[_key] = arguments[_key];
    }

    for (var _i = 0, _values = values; _i < _values.length; _i++) {
        var v = _values[_i];
        sum += v;
    }

    return sum;
};
```


### 可选catch参数

这是ES2019的特性，可以省略catch中的错误类型。在2018年上半年即被safari所支持。

```js 
let f3 = f2 => {
  try {
    f2();
  } catch {
    console.error("Error");
  }
};
```

转码之后，Babel会给我们生成一个未使用的error变量`_unused`：

```js 
var f1 = function f1(f2) {
    try {
        f2();
    } catch (_unused) {
        console.error("Error");
    }
};
```

带有error变量的情况下，v8通过CreateCatchContext为我们生成CatchContext，并且为catch块生成了一个CATCH_SCOPE:
``` 
         0x1937082936b6 @    0 : 19 ff fa          Mov <context>, r0
         0x1937082936b9 @    3 : 61 03 00          CallUndefinedReceiver0 a0, [0]
         0x1937082936bc @    6 : 8a 20             Jump [32] (0x1937082936dc @ 38)
         0x1937082936be @    8 : c3                Star1 
         0x1937082936bf @    9 : 82 f9 00          CreateCatchContext r1, [0]
         0x1937082936c2 @   12 : c4                Star0 
         0x1937082936c3 @   13 : 10                LdaTheHole 
         0x1937082936c4 @   14 : a6                SetPendingMessage 
         0x1937082936c5 @   15 : 0b fa             Ldar r0
         0x1937082936c7 @   17 : 1a f9             PushContext r1
         0x1937082936c9 @   19 : 21 01 02          LdaGlobal [1], [2]
         0x1937082936cc @   22 : c1                Star3 
         0x1937082936cd @   23 : 2d f7 02 04       LdaNamedProperty r3, [2], [4]
         0x1937082936d1 @   27 : c2                Star2 
         0x1937082936d2 @   28 : 13 03             LdaConstant [3]
         0x1937082936d4 @   30 : c0                Star4 
         0x1937082936d5 @   31 : 5e f8 f7 f6 06    CallProperty1 r2, r3, r4, [6]
         0x1937082936da @   36 : 1b f9             PopContext r1
         0x1937082936dc @   38 : 0e                LdaUndefined 
         0x1937082936dd @   39 : a9                Return 
Constant pool (size = 4)
0x19370829367d: [FixedArray] in OldSpace
 - map: 0x193708002205 <Map>
 - length: 4
           0: 0x193708293649 <ScopeInfo CATCH_SCOPE>
           1: 0x193708202741 <String[7]: #console>
           2: 0x193708202769 <String[5]: #error>
           3: 0x19370800455d <String[5]: #Error>
```

而对于可选catch参数的情况下，直接不生成CatchContext:
``` 
         0x19370829376a @    0 : 19 ff fa          Mov <context>, r0
         0x19370829376d @    3 : 61 03 00          CallUndefinedReceiver0 a0, [0]
         0x193708293770 @    6 : 8a 15             Jump [21] (0x193708293785 @ 27)
         0x193708293772 @    8 : 10                LdaTheHole 
         0x193708293773 @    9 : a6                SetPendingMessage 
         0x193708293774 @   10 : 21 00 02          LdaGlobal [0], [2]
         0x193708293777 @   13 : c2                Star2 
         0x193708293778 @   14 : 2d f8 01 04       LdaNamedProperty r2, [1], [4]
         0x19370829377c @   18 : c3                Star1 
         0x19370829377d @   19 : 13 02             LdaConstant [2]
         0x19370829377f @   21 : c1                Star3 
         0x193708293780 @   22 : 5e f9 f8 f7 06    CallProperty1 r1, r2, r3, [6]
         0x193708293785 @   27 : 0e                LdaUndefined 
         0x193708293786 @   28 : a9                Return 
Constant pool (size = 3)
0x193708293735: [FixedArray] in OldSpace
 - map: 0x193708002205 <Map>
 - length: 3
           0: 0x193708202741 <String[7]: #console>
           1: 0x193708202769 <String[5]: #error>
           2: 0x19370800455d <String[5]: #Error>
```

### Generator

解析赋值这样使用迭代器的方式是下一节转码效率高的部分。
但是，针对Generator这样显式使用迭代器的，就是另一番情况了。

我们看一个最简单的Generator，我们只生成几个数字：
```js 
let f1 = () => {
  let obj1 = {
    *[Symbol.iterator]() {
      yield 1;
      yield 2;
      yield 3;
    }
  };
  [...obj1];
};
```

我们看到，转码后的结果，不但定义了几个函数，还需要regeneratorRuntime运行时的支持：
```js 
function _toConsumableArray(arr) { return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread(); }

function _nonIterableSpread() { throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }

function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }

function _iterableToArray(iter) { if (typeof Symbol !== "undefined" && iter[Symbol.iterator] != null || iter["@@iterator"] != null) return Array.from(iter); }

function _arrayWithoutHoles(arr) { if (Array.isArray(arr)) return _arrayLikeToArray(arr); }

function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }

var f1 = function f1() {
  var obj1 = {
    [Symbol.iterator]() {
      return /*#__PURE__*/regeneratorRuntime.mark(function _callee() {
        return regeneratorRuntime.wrap(function _callee$(_context) {
          while (1) {
            switch (_context.prev = _context.next) {
              case 0:
                _context.next = 2;
                return 1;

              case 2:
                _context.next = 4;
                return 2;

              case 4:
                _context.next = 6;
                return 3;

              case 6:
              case "end":
                return _context.stop();
            }
          }
        }, _callee);
      })();
    }

  };

  _toConsumableArray(obj1);
};
```

### 类

虽然说class本质上跟Function的语法糖也差不多，但是，Babel转码生成出来的代码，可能比大多数同学想象的多。
我们看一个简单的例子：

```js 
  class Code {
    constructor(source) {
      this.source = source;
    }
  }
  code1 = new Code("test1.js");
```

转码之后效果如下，Babel为我们生成了_createClass,_classCallCheck和_defineProperties三个函数：

```js 
function _defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }

function _createClass(Constructor, protoProps, staticProps) { if (protoProps) _defineProperties(Constructor.prototype, protoProps); if (staticProps) _defineProperties(Constructor, staticProps); Object.defineProperty(Constructor, "prototype", { writable: false }); return Constructor; }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Code = /*#__PURE__*/_createClass(function Code(source) {
    _classCallCheck(this, Code);

    this.source = source;
});

code1 = new Code("test1.js");
```

## 转码效果更好

### 解构赋值

我们引用阮一峰老师的变量交换的的例子：

```js 
let f1 = () => {
  let x = 1;
  let y = 2;
  [x, y] = [y, x];
};

f1();
```

转码之后变成这样：

```js 
var f1 = function f1() {
  var x = 1;
  var y = 2;
  var _ref = [y, x];
  x = _ref[0];
  y = _ref[1];
};

f1();
```

我们先看下转码之后的字节码，有44个字节：
``` 
         0xf1208293682 @    0 : 0d 01             LdaSmi [1]
         0xf1208293684 @    2 : c4                Star0 
         0xf1208293685 @    3 : 0d 02             LdaSmi [2]
         0xf1208293687 @    5 : c3                Star1 
         0xf1208293688 @    6 : 79 00 00 25       CreateArrayLiteral [0], [0], #37
         0xf120829368c @   10 : c0                Star4 
         0xf120829368d @   11 : 0c                LdaZero 
         0xf120829368e @   12 : c1                Star3 
         0xf120829368f @   13 : 0b f9             Ldar r1
         0xf1208293691 @   15 : 36 f6 f7 01       StaInArrayLiteral r4, r3, [1]
         0xf1208293695 @   19 : 0d 01             LdaSmi [1]
         0xf1208293697 @   21 : c1                Star3 
         0xf1208293698 @   22 : 0b fa             Ldar r0
         0xf120829369a @   24 : 36 f6 f7 01       StaInArrayLiteral r4, r3, [1]
         0xf120829369e @   28 : 19 f6 f8          Mov r4, r2
         0xf12082936a1 @   31 : 0c                LdaZero 
         0xf12082936a2 @   32 : 2f f8 03          LdaKeyedProperty r2, [3]
         0xf12082936a5 @   35 : c4                Star0 
         0xf12082936a6 @   36 : 0d 01             LdaSmi [1]
         0xf12082936a8 @   38 : 2f f8 05          LdaKeyedProperty r2, [5]
         0xf12082936ab @   41 : c3                Star1 
         0xf12082936ac @   42 : 0e                LdaUndefined 
         0xf12082936ad @   43 : a9                Return
```

上面更简洁的解构需要多少个字节?
答案是189字节，因为涉及到迭代器：

```
         0xf120829376e @    0 : 0d 01             LdaSmi [1]
         0xf1208293770 @    2 : c4                Star0 
         0xf1208293771 @    3 : 0d 02             LdaSmi [2]
         0xf1208293773 @    5 : c3                Star1 
         0xf1208293774 @    6 : 79 00 00 25       CreateArrayLiteral [0], [0], #37
         0xf1208293778 @   10 : c1                Star3 
         0xf1208293779 @   11 : 0c                LdaZero 
         0xf120829377a @   12 : c2                Star2 
         0xf120829377b @   13 : 0b f9             Ldar r1
         0xf120829377d @   15 : 36 f7 f8 01       StaInArrayLiteral r3, r2, [1]
         0xf1208293781 @   19 : 0d 01             LdaSmi [1]
         0xf1208293783 @   21 : c2                Star2 
         0xf1208293784 @   22 : 0b fa             Ldar r0
         0xf1208293786 @   24 : 36 f7 f8 01       StaInArrayLiteral r3, r2, [1]
         0xf120829378a @   28 : b1 f7 03 05       GetIterator r3, [3], [5]
         0xf120829378e @   32 : 19 f7 f8          Mov r3, r2
         0xf1208293791 @   35 : 9f 07             JumpIfJSReceiver [7] (0xf1208293798 @ 42)
         0xf1208293793 @   37 : 65 bf 00 fa 00    CallRuntime [ThrowSymbolIteratorInvalid], r0-r0
         0xf1208293798 @   42 : c0                Star4 
         0xf1208293799 @   43 : 2d f6 01 07       LdaNamedProperty r4, [1], [7]
         0xf120829379d @   47 : c1                Star3 
         0xf120829379e @   48 : 12                LdaFalse 
         0xf120829379f @   49 : bf                Star5 
         0xf12082937a0 @   50 : 19 ff f2          Mov <context>, r8
         0xf12082937a3 @   53 : 0b f5             Ldar r5
         0xf12082937a5 @   55 : 96 21             JumpIfToBooleanTrue [33] (0xf12082937c6 @ 88)
         0xf12082937a7 @   57 : 11                LdaTrue 
         0xf12082937a8 @   58 : bf                Star5 
         0xf12082937a9 @   59 : 5d f7 f6 0d       CallProperty0 r3, r4, [13]
         0xf12082937ad @   63 : bb                Star9 
         0xf12082937ae @   64 : 9f 07             JumpIfJSReceiver [7] (0xf12082937b5 @ 71)
         0xf12082937b0 @   66 : 65 b7 00 f1 01    CallRuntime [ThrowIteratorResultNotAnObject], r9-r9
         0xf12082937b5 @   71 : 2d f1 02 0b       LdaNamedProperty r9, [2], [11]
         0xf12082937b9 @   75 : 96 0d             JumpIfToBooleanTrue [13] (0xf12082937c6 @ 88)
         0xf12082937bb @   77 : 2d f1 03 09       LdaNamedProperty r9, [3], [9]
         0xf12082937bf @   81 : bb                Star9 
         0xf12082937c0 @   82 : 12                LdaFalse 
         0xf12082937c1 @   83 : bf                Star5 
         0xf12082937c2 @   84 : 0b f1             Ldar r9
         0xf12082937c4 @   86 : 8a 03             Jump [3] (0xf12082937c7 @ 89)
         0xf12082937c6 @   88 : 0e                LdaUndefined 
         0xf12082937c7 @   89 : c4                Star0 
         0xf12082937c8 @   90 : 0b f5             Ldar r5
         0xf12082937ca @   92 : 96 21             JumpIfToBooleanTrue [33] (0xf12082937eb @ 125)
         0xf12082937cc @   94 : 11                LdaTrue 
         0xf12082937cd @   95 : bf                Star5 
         0xf12082937ce @   96 : 5d f7 f6 0f       CallProperty0 r3, r4, [15]
         0xf12082937d2 @  100 : bb                Star9 
         0xf12082937d3 @  101 : 9f 07             JumpIfJSReceiver [7] (0xf12082937da @ 108)
         0xf12082937d5 @  103 : 65 b7 00 f1 01    CallRuntime [ThrowIteratorResultNotAnObject], r9-r9
         0xf12082937da @  108 : 2d f1 02 0b       LdaNamedProperty r9, [2], [11]
         0xf12082937de @  112 : 96 0d             JumpIfToBooleanTrue [13] (0xf12082937eb @ 125)
         0xf12082937e0 @  114 : 2d f1 03 09       LdaNamedProperty r9, [3], [9]
         0xf12082937e4 @  118 : bb                Star9 
         0xf12082937e5 @  119 : 12                LdaFalse 
         0xf12082937e6 @  120 : bf                Star5 
         0xf12082937e7 @  121 : 0b f1             Ldar r9
         0xf12082937e9 @  123 : 8a 03             Jump [3] (0xf12082937ec @ 126)
         0xf12082937eb @  125 : 0e                LdaUndefined 
         0xf12082937ec @  126 : c3                Star1 
         0xf12082937ed @  127 : 0d ff             LdaSmi [-1]
         0xf12082937ef @  129 : bd                Star7 
         0xf12082937f0 @  130 : be                Star6 
         0xf12082937f1 @  131 : 8a 05             Jump [5] (0xf12082937f6 @ 136)
         0xf12082937f3 @  133 : bd                Star7 
         0xf12082937f4 @  134 : 0c                LdaZero 
         0xf12082937f5 @  135 : be                Star6 
         0xf12082937f6 @  136 : 10                LdaTheHole 
         0xf12082937f7 @  137 : a6                SetPendingMessage 
         0xf12082937f8 @  138 : bc                Star8 
         0xf12082937f9 @  139 : 0b f5             Ldar r5
         0xf12082937fb @  141 : 96 23             JumpIfToBooleanTrue [35] (0xf120829381e @ 176)
         0xf12082937fd @  143 : 19 ff f0          Mov <context>, r10
         0xf1208293800 @  146 : 2d f6 04 11       LdaNamedProperty r4, [4], [17]
         0xf1208293804 @  150 : 9e 1a             JumpIfUndefinedOrNull [26] (0xf120829381e @ 176)
         0xf1208293806 @  152 : b9                Star11 
         0xf1208293807 @  153 : 5d ef f6 13       CallProperty0 r11, r4, [19]
         0xf120829380b @  157 : 9f 13             JumpIfJSReceiver [19] (0xf120829381e @ 176)
         0xf120829380d @  159 : b8                Star12 
         0xf120829380e @  160 : 65 b7 00 ee 01    CallRuntime [ThrowIteratorResultNotAnObject], r12-r12
         0xf1208293813 @  165 : 8a 0b             Jump [11] (0xf120829381e @ 176)
         0xf1208293815 @  167 : ba                Star10 
         0xf1208293816 @  168 : 0c                LdaZero 
         0xf1208293817 @  169 : 1c f4             TestReferenceEqual r6
         0xf1208293819 @  171 : 98 05             JumpIfTrue [5] (0xf120829381e @ 176)
         0xf120829381b @  173 : 0b f0             Ldar r10
         0xf120829381d @  175 : a8                ReThrow 
         0xf120829381e @  176 : 0b f2             Ldar r8
         0xf1208293820 @  178 : a6                SetPendingMessage 
         0xf1208293821 @  179 : 0c                LdaZero 
         0xf1208293822 @  180 : 1c f4             TestReferenceEqual r6
         0xf1208293824 @  182 : 99 05             JumpIfFalse [5] (0xf1208293829 @ 187)
         0xf1208293826 @  184 : 0b f3             Ldar r7
         0xf1208293828 @  186 : a8                ReThrow 
         0xf1208293829 @  187 : 0e                LdaUndefined 
         0xf120829382a @  188 : a9                Return 
Constant pool (size = 5)
0xf1208293731: [FixedArray] in OldSpace
 - map: 0x0f1208002205 <Map>
 - length: 5
           0: 0x0f12082936fd <ArrayBoilerplateDescription PACKED_SMI_ELEMENTS, 0x0f12082936ed <FixedArray[2]>>
           1: 0x0f1208004e65 <String[4]: #next>
           2: 0x0f1208004441 <String[4]: #done>
           3: 0x0f1208005619 <String[5]: #value>
           4: 0x0f12080051dd <String[6]: #return>
```

## 兼容性还得再等一等

### Nullish运算符

Nullish运算符就是"??"运算符，如果为null和undefined则取"??"后边的值：

```js 
function greet(input) {
  return input ?? "Hello world";
}
```

翻译成字节码是9个字节：
```
         0x94c082935da @    0 : 0b 03             Ldar a0
         0x94c082935dc @    2 : 9e 04             JumpIfUndefinedOrNull [4] (0x94c082935e0 @ 6)
         0x94c082935de @    4 : 8a 04             Jump [4] (0x94c082935e2 @ 8)
         0x94c082935e0 @    6 : 13 00             LdaConstant [0]
         0x94c082935e2 @    8 : a9                Return 
```

转码之后的结果：
```js 
function greet(input) {
  return input !== null && input !== void 0 ? input : "Hello world";
}
```

翻译成字节码之后15个字节：
```
         0x2a8a082935da @    0 : 0b 03             Ldar a0
         0x2a8a082935dc @    2 : 9a 0a             JumpIfNull [10] (0x2a8a082935e6 @ 12)
         0x2a8a082935de @    4 : 0b 03             Ldar a0
         0x2a8a082935e0 @    6 : 9c 06             JumpIfUndefined [6] (0x2a8a082935e6 @ 12)
         0x2a8a082935e2 @    8 : 0b 03             Ldar a0
         0x2a8a082935e4 @   10 : 8a 04             Jump [4] (0x2a8a082935e8 @ 14)
         0x2a8a082935e6 @   12 : 13 00             LdaConstant [0]
         0x2a8a082935e8 @   14 : a9                Return
```

"??"运算符被v8译成JumpIfUndefinedOrNull指令，转码了之后就没有这待遇了，变成JumpIfNull和JumpIfUndefined两条指令。

所以，只要浏览器支持的话，Nullish运算符还是值得不转码的。

### 乘方运算符

与Nullish一样，乘方运算符也是有指令支持的。这样就节省了函数调用的开销。

```js 
let f1 = x => {
  return x ** x;
};

f1(10);
```

因为有Exp指令，一共6个字节就搞定了：
``` 
         0xb75082936be @    0 : 0b 03             Ldar a0
         0xb75082936c0 @    2 : 3e 03 00          Exp a0, [0]
         0xb75082936c3 @    5 : a9                Return
```

转码之后变成：
```js 
var f1 = function f1(x) {
  return Math.pow(x, x);
};

f1(10);
```

因为有函数调用，需要16个字节的指令：
``` 
         0xb7508293652 @    0 : 21 00 00          LdaGlobal [0], [0]
         0xb7508293655 @    3 : c3                Star1 
         0xb7508293656 @    4 : 2d f9 01 02       LdaNamedProperty r1, [1], [2]
         0xb750829365a @    8 : c4                Star0 
         0xb750829365b @    9 : 5f fa f9 03 03 04 CallProperty2 r0, r1, a0, a0, [4]
         0xb7508293661 @   15 : a9                Return 
Constant pool (size = 2)
0xb7508293621: [FixedArray] in OldSpace
 - map: 0x0b7508002205 <Map>
 - length: 2
           0: 0x0b75082028e5 <String[4]: #Math>
           1: 0x0b7508202aa1 <String[3]: #pow>
```

## 小结

从前面的案例中，我们可以看到，除了像解构引入了迭代这样的结构会变得复杂以外，大部分情况下，从源代码和字节码两个方面看，如果可以不转码，更有利于v8提升性能。
至少对于近几年的中高端机型上，值得我们使用新武器去取得更优的效果。
