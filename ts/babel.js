const babel = require("@babel/core");

const code = `
function add(x: number, y: number): number{
    return x+y;
}
`;

let ast0 = babel.transformSync(code, { ast: true, presets: ["@babel/preset-env", "@babel/preset-typescript"], filename:"index.ts", 
targets:"iOS 15" });
let str0 = ast0;
console.log(str0);