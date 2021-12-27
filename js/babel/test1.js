const code2 = 'import "@babel/core"; function greet(input) {return input ?? "Hello world";}';
const babel = require("@babel/core");
let result = babel.transformSync(code2, {
    targets: ">0.5%",
    presets: ["@babel/preset-env"]});

console.log(result.code);

let result2 = babel.transformSync(code2, {
    targets: ">1%",
    presets: ["@babel/preset-env"]});

console.log(result2.code);

const code3 = '';