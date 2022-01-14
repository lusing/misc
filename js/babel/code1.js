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

function process_code0(code) {
    let ast0 = babelParser.parse(code);
    console.log(ast0);
}

function process_code(code) {
    let ast0 = babelParser.parse(code);
    let program_body = ast0.program.body;
    for (let node1 of program_body){
        if(t.isExpressionStatement(node1)){
            console.log(node1.expression);
        }
    }
}

function process_code3(code) {
    let ast0 = babelParser.parse(code);
    traverse.default(ast0, {
        enter(path) {
            if(t.isProgram(path.node)){
                let program_body = path.node.body;
                for (let node1 of program_body){
                    console.log(node1);
                }
            }
            //console.log(path.node);
        }
    });
}


function process_code2(code) {
    let ast0 = babel.transformSync(code, { ast: true });
    traverse.default(ast0.ast, {
        enter(path) {
            console.log(path.node);
        }
    });

    //const c2 = generate.default(ast, {});
    //console.log(c2.code);
}