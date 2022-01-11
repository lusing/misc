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
    let ast0 = babel.transformSync(code, { ast: true, targets:"iOS 15" ,presets: ["@babel/preset-env"]});
    traverse.default(ast0.ast, {
        enter(path) {
            //if(t.isFunctionDeclaration(path.node) || t.isArrowFunctionExpression(path.node)){
            if(t.isVariableDeclaration(path.node)){
                let f1 = generate.default(path.node, {});
                console.log(f1.code);
                let code1 = f1.code;
                let ast1 = babel.transformSync(code1, { ast: true, targets:"iOS 9" ,presets: ["@babel/preset-env"]});
                console.log(ast1.code);
                console.log('=========');
            }
            //console.log(path.node);
        }
    });

    //const c2 = generate.default(ast, {});
    //console.log(c2.code);
}
