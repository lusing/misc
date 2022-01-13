# 用Babel操作JavaScript抽象语法树

前面我们通过《操作JavaScript的AST》一文，对于Babel操作JavaScript AST的强大功能有了感性的认识。

不过，细节是魔鬼。我们如果想深入地对js代码进行处理，还需要了解AST的细节。

## 如何从根节点开始遍历AST树

我们先搭建一个分析的框架。通过读取一个js文件，遍历每一个元素，然后将其打印出来。

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
    process_code(all_code);
}

function process_code(code) {
    let ast0 = babelParser.parse(code);
    console.log(ast0);
}
```

上面的文件假设叫code.js。然后我们们写一个最简单的，只有一个立即数的js文件，比如叫0.js
```js
0;
```

然后我们运行`node code.js 0.js`,解析出来的结果为:
```json
Node {
  type: 'File',
  start: 0,
  end: 2,
  loc: SourceLocation {
    start: Position { line: 1, column: 0 },
    end: Position { line: 1, column: 2 },
    filename: undefined,
    identifierName: undefined
  },
  errors: [],
  program: Node {
    type: 'Program',
    start: 0,
    end: 2,
    loc: SourceLocation {
      start: [Position],
      end: [Position],
      filename: undefined,
      identifierName: undefined
    },
    sourceType: 'script',
    interpreter: null,
    body: [ [Node] ],
    directives: []
  },
  comments: []
}
```

因为我们是从文件中解析出来的，所以根节点是一个File节点。
然后File节点中包含了一个Program节点。
Program节点中会包含一个body数组，其中保存了程序的每一个节点。
比如我们只有0这一条语句，所以就只有一个节点。

我们再给其增加一行：
```js
0;
-0.0;
```

我们可以看到，body变成了两个节点：
```json
body: [ [Node], [Node] ],
```

我们就可以通过for of循环去遍历程序节点：
```js
function process_code(code) {
    let ast0 = babelParser.parse(code);
    let program_body = ast0.program.body;
    for (let node1 of program_body){
        console.log(node1);
    }
}
```

输出结果是两个ExpressionStatement：

```json
Node {
  type: 'ExpressionStatement',
  start: 0,
  end: 2,
  loc: SourceLocation {
    start: Position { line: 1, column: 0 },
    end: Position { line: 1, column: 2 },
    filename: undefined,
    identifierName: undefined
  },
  expression: Node {
    type: 'NumericLiteral',
    start: 0,
    end: 1,
    loc: SourceLocation {
      start: [Position],
      end: [Position],
      filename: undefined,
      identifierName: undefined
    },
    extra: { rawValue: 0, raw: '0' },
    value: 0
  }
}
Node {
  type: 'ExpressionStatement',
  start: 3,
  end: 8,
  loc: SourceLocation {
    start: Position { line: 2, column: 0 },
    end: Position { line: 2, column: 5 },
    filename: undefined,
    identifierName: undefined
  },
  expression: Node {
    type: 'UnaryExpression',
    start: 3,
    end: 7,
    loc: SourceLocation {
      start: [Position],
      end: [Position],
      filename: undefined,
      identifierName: undefined
    },
    operator: '-',
    prefix: true,
    argument: Node {
      type: 'NumericLiteral',
      start: 4,
      end: 7,
      loc: [SourceLocation],
      extra: [Object],
      value: 0
    }
  }
}
```

## 表达式语句

表达式语句是一个容器，将表达式内容封装在ExpressionStatement节点的expression字段里面。

我们可以通过@babel/types的isExpressionStatement函数来判断一个节点是不是表达式节点：
```js
function process_code(code) {
    let ast0 = babelParser.parse(code);
    let program_body = ast0.program.body;
    for (let node1 of program_body){
        if(t.isExpressionStatement(node1)){
            console.log(node1.expression);
        }
    }
}
```

### 立即数

对于立即数，它只有一个值，没有更复杂的结构，比如0:
```js
Node {
  type: 'NumericLiteral',
  start: 0,
  end: 1,
  loc: SourceLocation {
    start: Position { line: 1, column: 0 },
    end: Position { line: 1, column: 1 },
    filename: undefined,
    identifierName: undefined
  },
  extra: { rawValue: 0, raw: '0' },
  value: 0
}
```

再比如字符串类型的"Hello":
```js
Node {
  type: 'StringLiteral',
  start: 3,
  end: 10,
  loc: SourceLocation {
    start: Position { line: 2, column: 0 },
    end: Position { line: 2, column: 7 },
    filename: undefined,
    identifierName: undefined
  },
  extra: { rawValue: 'Hello', raw: '"Hello"' },
  value: 'Hello'
}
```

又如长整数，123n:
```js
Node {
  type: 'BigIntLiteral',
  start: 12,
  end: 16,
  loc: SourceLocation {
    start: Position { line: 3, column: 0 },
    end: Position { line: 3, column: 4 },
    filename: undefined,
    identifierName: undefined
  },
  extra: { rawValue: '123', raw: '123n' },
  value: '123'
}
```
