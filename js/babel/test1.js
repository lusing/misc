const code2 = 'function greet(input) {return input ?? "Hello world";}; greet();';
const babel = require("@babel/core");

function generate_codes(code){
    let result = babel.transformSync(code, {
        targets: ">0.5%",
        presets: ["@babel/preset-env"]});
    
    console.log(result.code);
    
    let result2 = babel.transformSync(code, {
        targets: ">1%",
        presets: ["@babel/preset-env"]});
    
    console.log(result2.code);
}

generate_codes(code2);

const code3 = 'let [a,b,c]=[1,2,3];';
generate_codes(code3);

const code4 = 'let f1 = () => {let a=0; a++;}; f1();'
generate_codes(code4);



const code5 = `
let f1 = () => {
    let sum = 0;
    for(let i=0;i<10;i++){
        sum += i;
    }
}
f1();
`;
generate_codes(code5);

const code6 = 'let f1 = () => {const a = 0; a=2;};'
generate_codes(code6);

const code7 = 'console.log(this);'
generate_codes(code7);

const code8 = 'let f1 = () => {let [a=1]=[void 0];};f1();';
generate_codes(code8);

const code9 = `
let f1 = () => {
    let x = 1;
    let y = 2;
    [x,y] = [y,x];
}
f1();
`;
generate_codes(code9);

const code10 = "let f1 = () => { String.raw`\n`;};f1();";
generate_codes(code10);
