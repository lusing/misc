# 操作JavaScript的AST

前面我们学习了eslint和stylelint的规则的写法，当大家实际去写的时候，一定会遇到很多细节的问题，比如解析的代码是有错误的，或者是属性值不足以分析出问题来之类的。我们还需要更多的工具来帮助我们简化规则开发的流程。比如说容错度更高的解析器，或者获取更丰富属性的工具。

我们知道，Eslint主要是基于AST层次进行操作的。

我们知道，eslint支持更换解析器，那么，它就需要一套标准。eslint使用的这套标准叫做estree规范。estree规范的指导委员会的三名成员，恰好来自eslint, acorn和babel.

Estree的基础格式，可以从[ES5规范](https://github.com/estree/estree/blob/master/es5.md)中查看到，从[ES6规范](https://github.com/estree/estree/blob/master/es2015.md)开始，每个版本都增加新的规范。比如[ES2016](https://github.com/estree/estree/blob/master/es2016.md)增加对"**"运算符的支持。

acorn解析器是支持plugin机制的，于是eslint所用的espree解析器和babel的解析器就都在acorn上进行扩展。



## Acorn解析器



acorn的默认用法非常简单，直接来段代码字符串parse一下就出来AST结构了：
```js
let acorn = require("acorn");

console.log(acorn.parse("for(let i=0;i<10;i+=1){console.log(i);}", {ecmaVersion: 2020}));
```

输出如下：
```
Node {
  type: 'Program',
  start: 0,
  end: 39,
  body: [
    Node {
      type: 'ForStatement',
      start: 0,
      end: 39,
      init: [Node],
      test: [Node],
      update: [Node],
      body: [Node]
    }
  ],
  sourceType: 'script'
}
```



### 遍历语法树



解析好了语法树节点之后，我们就可以遍历树了。acorn-walk包为我们提供了遍历的能力。

Acorn-walk提供了几种粒度的遍历方式，比如我们用simple函数遍历所有的Literal值：

```js
const acorn = require("acorn")
const walk = require("acorn-walk")

const code = 'for(let i=0;i<10;i+=1){console.log(i);}';

walk.simple(acorn.parse(code, {ecmaVersion:2020}), {
    Literal(node) {
        console.log(`Found a literal: ${node.value}`);
    }
});
```

输出如下：

```json
Found a literal: 0
Found a literal: 10
Found a literal: 1
```



当然，更经常使用的是full函数：

```js
const acorn = require("acorn")
const walk = require("acorn-walk")

const code = 'for(let i=0;i<10;i+=1){console.log(i);}';
const ast1 = acorn.parse(code, {ecmaVersion:2020});

walk.full(ast1, function(node){
    console.log(node.type);
});
```



输出如下：

```json
Identifier
Literal
VariableDeclarator
VariableDeclaration
Identifier
Literal
BinaryExpression
Identifier
Literal
AssignmentExpression
Identifier
MemberExpression
Identifier
CallExpression
ExpressionStatement
BlockStatement
ForStatement
Program
```



我们可以看到，最后是树根Program.



### 高容错版本 acorn-loose



Acorn正常使用起来没有什么问题，但是还有一点可以做得更好，就是容错的情况。

我们看一个出错的例子：

```js
let acorn = require("acorn");

console.log(acorn.parse("let a = 1 );", {ecmaVersion: 2020}));
```



Acorn就不干了，报错：

```json
SyntaxError: Unexpected token (1:10)
    at Parser.pp$4.raise (acorn/node_modules/acorn/dist/acorn.js:3434:15)
    at Parser.pp$9.unexpected (acorn/node_modules/acorn/dist/acorn.js:749:10)
    at Parser.pp$9.semicolon (acorn/node_modules/acorn/dist/acorn.js:726:68)
    at Parser.pp$8.parseVarStatement (acorn/node_modules/acorn/dist/acorn.js:1157:10)
    at Parser.pp$8.parseStatement (acorn/node_modules/acorn/dist/acorn.js:904:19)
    at Parser.pp$8.parseTopLevel (acorn/node_modules/acorn/dist/acorn.js:806:23)
    at Parser.parse (acorn/node_modules/acorn/dist/acorn.js:579:17)
    at Function.parse (acorn/node_modules/acorn/dist/acorn.js:629:37)
    at Object.parse (acorn/node_modules/acorn/dist/acorn.js:5546:19)
    at Object.<anonymous> (acorn/normal.js:3:19) {
  pos: 10,
  loc: Position { line: 1, column: 10 },
  raisedAt: 11
}
```



下面我们换成高容错版本的acorn-loose:

```js
let acornLoose = require("acorn-loose");

console.log(acornLoose.parse("let a = 1 );", { ecmaVersion: 2020 }));
```



Acorn-loose会将我们多写的半个括号识别成一个空语句：

```json
Node {
  type: 'Program',
  start: 0,
  end: 12,
  body: [
    Node {
      type: 'VariableDeclaration',
      start: 0,
      end: 9,
      kind: 'let',
      declarations: [Array]
    },
    Node { type: 'EmptyStatement', start: 11, end: 12 }
  ],
  sourceType: 'script'
}
```



## Espree解析器



espree既然是扩展acorn，基本用法当然是兼容的:

```js
const espree = require("espree");

const code = "for(let i=0;i<10;i+=1){console.log(i);}";

const ast = espree.parse(code,{ ecmaVersion: 2020 });

console.log(ast);
```

生成的格式当然也是estree, 跟acorn一样：
```
Node {
  type: 'Program',
  start: 0,
  end: 39,
  body: [
    Node {
      type: 'ForStatement',
      start: 0,
      end: 39,
      init: [Node],
      test: [Node],
      update: [Node],
      body: [Node]
    }
  ],
  sourceType: 'script'
}
```

如果看AST还不行，我们还可以直接看分词的效果：
```js
const tokens = espree.tokenize(code,{ ecmaVersion: 2020 });
console.log(tokens);
```
结果如下：
```
[
  Token { type: 'Keyword', value: 'for', start: 0, end: 3 },
  Token { type: 'Punctuator', value: '(', start: 3, end: 4 },
  Token { type: 'Keyword', value: 'let', start: 4, end: 7 },
  Token { type: 'Identifier', value: 'i', start: 8, end: 9 },
  Token { type: 'Punctuator', value: '=', start: 9, end: 10 },
  Token { type: 'Numeric', value: '0', start: 10, end: 11 },
  Token { type: 'Punctuator', value: ';', start: 11, end: 12 },
  Token { type: 'Identifier', value: 'i', start: 12, end: 13 },
  Token { type: 'Punctuator', value: '<', start: 13, end: 14 },
  Token { type: 'Numeric', value: '10', start: 14, end: 16 },
  Token { type: 'Punctuator', value: ';', start: 16, end: 17 },
  Token { type: 'Identifier', value: 'i', start: 17, end: 18 },
  Token { type: 'Punctuator', value: '+=', start: 18, end: 20 },
  Token { type: 'Numeric', value: '1', start: 20, end: 21 },
  Token { type: 'Punctuator', value: ')', start: 21, end: 22 },
  Token { type: 'Punctuator', value: '{', start: 22, end: 23 },
  Token { type: 'Identifier', value: 'console', start: 23, end: 30 },
  Token { type: 'Punctuator', value: '.', start: 30, end: 31 },
  Token { type: 'Identifier', value: 'log', start: 31, end: 34 },
  Token { type: 'Punctuator', value: '(', start: 34, end: 35 },
  Token { type: 'Identifier', value: 'i', start: 35, end: 36 },
  Token { type: 'Punctuator', value: ')', start: 36, end: 37 },
  Token { type: 'Punctuator', value: ';', start: 37, end: 38 },
  Token { type: 'Punctuator', value: '}', start: 38, end: 39 }
]
```



从结果可以看到，词法分析之后的结果是token，而语法分析之后的已经是语句的结果了。

关于为什么eslint为什么在acorn之上还要封装一个espree，是因为最早eslint依赖于esprima，二者之间有不兼容的地方，eslint需要更多的信息来分析代码。



## Babel基础操作



最后出场的是虽然不是eslint默认，但是双向支持都不错的重型武器babel.



### Babel解析器



大杀器babel也可以配置成只要ast的模式：

```js
const code2 = 'function greet(input) {return input ?? "Hello world";}';

const babel = require("@babel/core");
result = babel.transformSync(code2, { ast: true });

console.log(result.ast);
```

输出结果是这样的：
```
Node {
  type: 'File',
  start: 0,
  end: 54,
  loc: SourceLocation {
    start: Position { line: 1, column: 0 },
    end: Position { line: 1, column: 54 },
    filename: undefined,
    identifierName: undefined
  },
  errors: [],
  program: Node {
    type: 'Program',
    start: 0,
    end: 54,
    loc: SourceLocation {
      start: [Position],
      end: [Position],
      filename: undefined,
      identifierName: undefined
    },
    sourceType: 'module',
    interpreter: null,
    body: [ [Node] ],
    directives: [],
    leadingComments: undefined,
    innerComments: undefined,
    trailingComments: undefined
  },
  comments: [],
  leadingComments: undefined,
  innerComments: undefined,
  trailingComments: undefined
}
```

我们还可以用babel.parseSync方法只去读取AST:
```js
result2 = babel.parseSync(code2);
console.log(result2);
```

甚至我们可以只用parser包：
```js
const babelParser = require('@babel/parser');
console.log(babelParser.parse(code2, {}));
```



### Babel的遍历器



Acorn有专门的遍历器包，Babel当然也不甘示弱，提供了@babel/traverse包来辅助遍历抽象语法树。



我们来看个代码节点路径的例子：

```js
const code4 = 'let a = 2 ** 8;'
const ast4 = babelParser.parse(code4, {})
const traverse2 = require("@babel/traverse");
traverse2.default(ast4, {
    enter(path) {
        console.log(path.type);
    }
});
```

输出如下，是从Program自顶向下的路径：

```json
Program
VariableDeclaration
VariableDeclarator
Identifier
BinaryExpression
NumericLiteral
NumericLiteral
```



### 类型判断



遍历之后，我们需要大量的工具函数去进行类型判断。Babel给我们提供了一个巨大的工具类库@babel/types.

比如，我们想判断一个AST节点是不是标识符，就可以调用isIdentifier函数去判断下，我们看个例子：

```js
const code6 = 'if (a==2) {a+=1};';
const t = require('@babel/types');
const ast6 = babelParser.parse(code6, {})
traverse2.default(ast6, {
    enter(path) {
        if (t.isIdentifier(path.node)) {
            console.log(path.node);
        }
    }
});
```



输出如下：

```json
Node {
  type: 'Identifier',
  start: 4,
  end: 5,
  loc: SourceLocation {
    start: Position { line: 1, column: 4 },
    end: Position { line: 1, column: 5 },
    filename: undefined,
    identifierName: 'a'
  },
  name: 'a'
}
Node {
  type: 'Identifier',
  start: 11,
  end: 12,
  loc: SourceLocation {
    start: Position { line: 1, column: 11 },
    end: Position { line: 1, column: 12 },
    filename: undefined,
    identifierName: 'a'
  },
  name: 'a'
}
```



现在，我们要判断有没有表达式使用了"=="运算符，就可以这样写：

```js
const code8 = 'if (a==2) {a+=1};';
const ast8 = babelParser.parse(code6, {})
traverse2.default(ast8, {
    enter(path) {
        if (t.isBinaryExpression(path.node)) {
            if(path.node.operator==="=="){
                console.log(path.node);
            }
        }
    }
});
```

isBinaryExpression也支持参数，我们可以把运算符的条件加上：

```js
traverse2.default(ast8, {
    enter(path) {
        if (t.isBinaryExpression(path.node,{operator:"=="})) {
            console.log(path.node);
        }
    }
});
```



### 构造AST节点



光能判断类型还不算什么。@babel/type库的更主要的作用是可以用来生成AST Node。

比如我们要生成一个二元表达式，就可以用binaryExpression函数来生成：

```js
let node7 = t.binaryExpression("==",t.identifier("a"),t.numericLiteral(0));
console.log(node7);
```

注意，标识符和字面量都不能生接给值，而是要用自己类型的构造函数来生成哈。

运行结果如下：

```
{
  type: 'BinaryExpression',
  operator: '==',
  left: { type: 'Identifier', name: 'a' },
  right: { type: 'NumericLiteral', value: 0 }
}
```



要把运算符"=="改成"==="，直接替换掉就好：

```js
node7.operator="===";
console.log(node7);
```

输出结果如下：

```
{
  type: 'BinaryExpression',
  operator: '===',
  left: { type: 'Identifier', name: 'a' },
  right: { type: 'NumericLiteral', value: 0 }
}
```



我们把上面的逻辑串一下，将"=="运算符替换成"==="运算符的代码如下：

```js
const code8 = 'if (a==2) {a+=1};';
const ast8 = babelParser.parse(code6, {})
traverse2.default(ast8, {
    enter(path) {
        if (t.isBinaryExpression(path.node,{operator:"=="})) {
            path.node.operator = "===";
        }
    }
});
```



### AST生成代码



下面我们的高光时刻来了，直接生成代码。babel为我们准备了"@babel/generator"包：

```js
const generate = require("@babel/generator") ;
let c2 = generate.default(ast8,{});
console.log(c2.code);
```



生成的代码如下：

```js
if (a === 2) {
  a += 1;
}

;
```



### 代码模板



我们要生成的代码都通过AST表达式来写有时候有点反人性，这时候我们可以尝试下代码模板。



我们来看个例子：

```js
const babelTemplate = require("@babel/template");
const requireTemplate = babelTemplate.default(`
  const IMPORT_NAME = require(SOURCE);
`);

const ast9 = requireTemplate({
    IMPORT_NAME: t.identifier("babelTemplate"),
    SOURCE: t.stringLiteral("@babel/template")
});

console.log(ast9);
```



请注意，通过代码模板生成的直接就是AST哈，做的可不是模板字符串替换，替换的可是标识符和文本字面常量。

输出结果如下：

```json
{
  type: 'VariableDeclaration',
  kind: 'const',
  declarations: [
    {
      type: 'VariableDeclarator',
      id: [Object],
      init: [Object],
      loc: undefined
    }
  ],
  loc: undefined
}
```



想要转成源代码，还需要调用generate包：

```js
console.log(generate.default(ast9).code);
```

输出如下：

```json
const babelTemplate = require("@babel/template");
```



另外，需要注意的是，我们的代码模板生成的是抽象语法树，不是具体语法树，比如我们在代码模板里写了注释，最后生成回代码里可就没有了：

```js
const forTemplate = babelTemplate.default(`
    for(let i=0;i<END;i+=1){
        console.log(i); // output loop variable
    }
`);
const ast10 = forTemplate({
    END: t.numericLiteral(10)
});
console.log(generate.default(ast10).code);
```



生成的代码如下：

```js
for (let i = 0; i < 10; i += 1) {
  console.log(i);
}
```



## Babel高级操作

### Babel转码器

既然有了babel，我们只用其parser有点浪费了，我们可以在我们的代码中使用babel来作为转码器：

```js
const code2 = 'function greet(input) {return input ?? "Hello world";}';
const babel = require("@babel/core");
let result = babel.transformSync(code2, {
    targets: ">0.5%",
    presets: ["@babel/preset-env"]});

console.log(result.code);
```

记得安装@babel/core和@babel/preset-env。

结果如下：

```js
"use strict";

function greet(input) {
  return input !== null && input !== void 0 ? input : "Hello world";
}
```

我们再来个ES 6 Class转换的例子：

```js
const code3 = `
//Test Class Function
class Test {
    constructor() {
      this.x = 2;
    }
  }`;

const babel = require("@babel/core");
let result = babel.transformSync(code3, {
    presets: ["@babel/preset-env"]
});

console.log(result.code);
```

除了presets: ["@babel/preset-env"]需要指定外，其它用缺省的参数就好。

生成的代码如下：

```js
"use strict";

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

//Test Class Function
var Test = function Test() {
  _classCallCheck(this, Test);

  this.x = 2;
};
```



在eslint规则中，如果源代码没有转码器，我们就可以利用babel直接转码生成autofix.



### AST节点的替换



前面我们只修改了二元表达式中的运算符，不过这样的情况在实际中很少见。实际情况中我们经常要修改一大段表达式。这时候我们可以用replaceWith函数将旧的AST节点换成新的AST节点。

还以将"=="换成"==="为例，这次我们改成直接生成一个新的binaryExpression来替换原有的，表达式中的左右节点都不变：

```js
const babel = require("@babel/core");
const babelParser = require('@babel/parser');
const t = require('@babel/types');
const traverse = require("@babel/traverse");
const generate = require("@babel/generator");

const code8 = 'if (a==2) {a+=1}; if (a!=0) {a=0}';
const ast8 = babelParser.parse(code8, {})
traverse.default(ast8, {
    enter(path) {
        if (t.isBinaryExpression(path.node, {operator: "=="})) {
            path.replaceWith(t.binaryExpression("===", path.node.left, path.node.right));
        }else if(t.isBinaryExpression(path.node, {operator: "!="})){
            path.replaceWith(t.binaryExpression("!==", path.node.left, path.node.right));
        }
    }
});

let c2 = generate.default(ast8, {});
console.log(c2.code);
```



输出结果如下：

```js
if (a === 2) {
  a += 1;
}

;

if (a !== 0) {
  a = 0;
}
```



### AST节点的删除



我们在review代码的时候，经常有发现console.log语句没有被删除之类的问题。此时我们就可以写一个AST处理工具，将console.log语句删除。直接调用节点的remove方法就可以删除掉当前节点。

console.log是一个函数调用，它是一个CallExpression，调用者是CallExpression的callee属性：

```js
let code11 = "let a = 1; console.log(a);"
const ast11 = babelParser.parse(code11, {})
traverse.default(ast11, {
    enter(path) {
        if (t.isCallExpression(path) && t.isMemberExpression(path.node.callee)) {
            if (path.node.callee.object.name === "console" && path.node.callee.property.name === "log") {
                path.remove();
            }
        }
    }
});
const c11 = generate.default(ast11, {});
console.log(c11.code);
```



输出如下：

```js
let a = 1;
```



或者干脆更进一步，只要是console对象，管它调用的是什么函数，统统都删掉：

```js
let code12 = "let a = 1; console.log(a); console.info('Hello,World!')";
const ast12 = babelParser.parse(code12, {})
traverse.default(ast12, {
    enter(path) {
        if (t.isCallExpression(path) && t.isMemberExpression(path.node.callee)) {
            if (path.node.callee.object.name === "console") {
                path.remove();
            }
        }
    }
});
const c12 = generate.default(ast11, {});
console.log(c12.code);
```



### 作用域



Babel同样支持作用域的分析。

比如，我们可以用scope.hasBinding检查在这个scope中某局部变量是否有被绑定，也可以用scope.hasGlobal来检查是否定义了某全局变量。

如果本作用域有绑定变量，可以通过getBinding函数来获取其初始值。



我们来看个例子：

```js
let code13 = `
g = 1;
function test(){
    let a = 0;
    for(let i = 0;i<10;i++){
        a+=i;
    }
}
`;
const ast13 = babelParser.parse(code13, {})
traverse.default(ast13, {
    enter(path) {
        console.log(path.type);
        const is_a = path.scope.hasBinding('a');
        console.log(is_a);
        if(is_a){
            console.log(path.scope.getBinding('a').path.node.init.value);
        }
        console.log(path.scope.hasGlobal('g'));
    }
});
```



输出如下：

```json
Program
false
true
ExpressionStatement
false
true
AssignmentExpression
false
true
Identifier
false
true
NumericLiteral
false
true
FunctionDeclaration
true
0
true
Identifier
true
0
true
BlockStatement
true
0
true
...
```



我们看到，到了函数声明FunctionDeclaration开始，函数内定义的变量a开始被绑定，我们能够获取到其初始值0.



### 用Babel高亮和标记出错代码



除了可以分析修改AST、AST生成代码和转码这些常规操作之外，Babel还提供了code-frame功能来标记代码，让出错信息的可读性更好。



我们来看个例子：

```js
const codeFrame = require("@babel/code-frame");
const rawLines2 = 'let a = isNaN(b);';
const result2 = codeFrame.codeFrameColumns(rawLines2, {
    start: {line: 1, column: 9},
    end: {line: 1, column: 14},
}, {highlightCode: true});

console.log(result2);
```



我们来看下结果：

![](https://gw.alicdn.com/imgextra/i2/O1CN014xB9U326HnToClYcy_!!6000000007637-2-tps-396-92.png)

有代码高亮，还有错误标红，是不是对用户很友好？



我们再看个跨行的例子，我们只要标记首尾信息就好，其余交给@babel/code-frame去解决：

```js
const rawLines3 = ["class CodeAnalyzer {", "  constructor()", "};"].join("\n");
const result3 = codeFrame.codeFrameColumns(rawLines3, {
    start: {line: 2, column: 3},
    end: {line: 2, column: 16},
}, {highlightCode: true});

console.log(result3);
```



输出结果如下：

![](https://gw.alicdn.com/imgextra/i2/O1CN01qrYoIb1zTWIyzeYCt_!!6000000006715-2-tps-450-184.png)

## 将isNaN替换为Number.isNaN的完整例子

上面的知识可能还有点零散，我们来个例子将它们串一下。
下面的js脚本从命令行参数读入一个js文件名，然后去查找它的isNaN的调用，主要是要把参数保存起来，接着将其替换为Number.isNaN的调用：


```js 
const babel = require("@babel/core");
const babelParser = require('@babel/parser');
const t = require('@babel/types');
const traverse = require("@babel/traverse");
const generate = require("@babel/generator");
const babelTemplate = require("@babel/template");
const fs = require("fs");

let args = process.argv;

if (args.length !== 3) {
    console.error('Please input a js file name:');
} else {
    let filename = args[2];
    let all_code = fs.readFileSync(filename, { encoding: 'utf8' });
    fix(all_code);
}

function fix(code) {
    const isNaNTemplate = babelTemplate.default(`Number.isNaN(ARG);`);
    const ast0 = babel.transformSync(code, { ast: true })?.ast;
    traverse.default(ast0, {
        enter(path) {
            if (t.isCallExpression(path) && path.node.callee.name === 'isNaN') {
                let arg1 = path.node.arguments;
                const node2 = isNaNTemplate({
                    ARG: arg1,
                });
                path.replaceWith(node2);
            }
        }
    });

    const c2 = generate.default(ast0, {});
    console.log(c2.code);
}
```


## 小结

通过本文学习这些工具，我们扩展了解析高容错代码的能力，修改和生成AST的能力，重新生成代码的能力等。AST级别的操作比起直接处理源代码和文本替换，不管是在安全性还是便利上都有经较显著的提升。

另外，我们也可以查看v8生成的AST，具体请参看：[将v8变成工具](https://ata.alibaba-inc.com/articles/222218)。


