const babel = require("@babel/core");
const babelParser = require('@babel/parser');
const t = require('@babel/types');
const traverse = require("@babel/traverse");
const generate = require("@babel/generator");
const babelTemplate = require("@babel/template");
const fs = require("fs");

const file_6 = "./es6.js";
const file_5 = "./es5.js";
const file_6_full = "./es6_full.js";
const file_5_full = "./es5_full.js";

let args = process.argv;

if (args.length !== 3) {
    console.error('Please input a js file name:');
} else {
    let filename = args[2];
    let all_code = fs.readFileSync(filename, { encoding: 'utf8' });
    process_code(all_code);
}



function process_code(code) {
    let ast0 = babel.transformSync(code, { ast: true, targets:"iOS 15" ,presets: ["@babel/preset-env"]});
    traverse.default(ast0.ast, {
        enter(path) {
            //if(t.isFunctionDeclaration(path.node) || t.isArrowFunctionExpression(path.node)){
            if(t.isVariableDeclaration(path.node)){
                let f1 = generate.default(path.node, {});
                console.log(f1.code);
                let code_6 = f1.code.replace(/[\n\t]/g, '');
                fs.appendFileSync(file_6,code_6);fs.appendFileSync(file_6,'\n');
                fs.appendFileSync(file_6_full,f1.code);fs.appendFileSync(file_6_full,'\n==========\n');
                let code1 = f1.code;
                let ast1 = babel.transformSync(code1, { ast: true, targets:"iOS 9" ,presets: ["@babel/preset-env"]});
                console.log(ast1.code);
                let code_5 = ast1.code.replace(/[\n\t]/g, '');
                fs.appendFileSync(file_5,code_5);fs.appendFileSync(file_5,'\n');
                fs.appendFileSync(file_6_full,ast1.code);fs.appendFileSync(file_5_full,'\n==========\n');
                console.log('=========');
            }
            //console.log(path.node);
        }
    });

    //const c2 = generate.default(ast, {});
    //console.log(c2.code);
}
