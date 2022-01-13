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
    remove_function_comments(all_code);
}



function remove_function_comments(code) {
    let ast0 = babel.transformSync(code, { ast: true, targets:"iOS 15" ,presets: ["@babel/preset-env"]});
    traverse.default(ast0.ast, {
        enter(path) {
            if((path.node.type ==="FunctionDeclaration")){
                path.node.leadingComments=[];
            }
            //console.log(path.node);
        }
    });

    const c2 = generate.default(ast0.ast, {targets:"iOS 15" ,presets: ["@babel/preset-env"]});
    console.log(c2.code);
}
