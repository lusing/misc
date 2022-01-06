const code2 = 'function greet(input) {return input ?? "Hello world";}; greet();';
const babel = require("@babel/core");

function generate_codes(code) {
    let result = babel.transformSync(code, {
        targets: ">0.1%",
        sourceMaps: true,
        presets: ["@babel/preset-env"]
    });

    console.log(result.code);
    console.log(result.map);

    let result2 = babel.transformSync(code, {
        targets: ">1%",
        sourceMaps: "both",
        presets: ["@babel/preset-env"]
    });

    console.log(result2.code);

    let result3 = babel.transformSync(code, {
        targets: "iOS 9",
        sourceMaps: true,
        presets: ["@babel/preset-env"]
    });

    console.log(result3.code);
    console.log('==========');
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

const code11 = `let f1 = () => {
     let a1 = 100_000_000;
     let a2 = 100_000_000n;
     let a3 = BigInt(100);
};f1();`;
generate_codes(code11);

const code12 = `let f1 = () => {
    Number.isNaN(NaN);
    Number.isNaN(undefined);
};f1();`;
generate_codes(code12);

const code13 = `let f1 = (x) => {
    return x ** x;
};f1(10);`;
generate_codes(code13);

const code14 = `let f1 = (...values) => {
    let sum = 0;
    for(let v of values) {
        sum += v;
    }
    return sum;    
};f1(1,4,9);`;
generate_codes(code14);

const code15 = `let f1 = (f2) => {
    try{
        f2();
    }catch{
        console.error("Error");
    }
};f1(console.log);`;
generate_codes(code15);

const code16 = `let f1 = () => {
    const a1 = [1,2,3,4,5,6,7,8,9,10];
    let a2 = [...a1];
};
f1();`;
generate_codes(code16);

const code17 = `let f1 = () => {
    let s1 = Symbol();
    return typeof s1;
};
f1();`;
generate_codes(code17);

const code18 = `let f1 = () => {
    const s1 = new Set();
    [2,3,4,5,5,5].forEach ( x=> s1.add(x));
};
f1();`;
generate_codes(code18);


const code19 = `let f1 = () => {
    let obj1 = {
        *[Symbol.iterator](){
            yield 1;
            yield 2;
            yield 3;
        }
    };
    [...obj1];
};
f1();`;
generate_codes(code19);

class Code{
    constructor(source){
        this.source = source;
    }
}

const code20 = `let f1 = () => {
    class Code{
        constructor(source){
            this.source = source;
        }
    }
    code1 = new Code("test1.js");    
};
f1();`;
generate_codes(code20);
