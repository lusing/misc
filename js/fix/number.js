const babel = require("@babel/core");
const babelParser = require('@babel/parser');
const t = require('@babel/types');
const traverse = require("@babel/traverse");
const generate = require("@babel/generator");
const babelTemplate = require("@babel/template");

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
    const ast = babelParser.parse(code, {})
    traverse.default(ast, {
        enter(path) {
            if (t.isCallExpression(path) && path.node.callee.name === 'isNaN') {
                console.log(path.node);
                let arg1 = path.node.arguments;
                const node2 = isNaNTemplate({
                    ARG: arg1,
                });
                console.log(node2);
                path.replaceWith(node2);
            }
        }
    });

    const c2 = generate.default(ast, {});
    console.log(c2.code);
}

