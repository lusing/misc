import transformSync from "@babel/core";

function add(x: number, y: number): number{
    return x+y;
}
;
let f1 = (x: bigint, y: bigint): bigint => x+y;
f1(10n,20n);

const code:string = `
function add(x: number, y: number): number{
    return x+y;
}
`;

let ast0 = transformSync(code, { ast: true, presets: ["@babel/preset-env", "@babel/preset-react"] });
let str0 = ast0.code;//.replace(/[\n\t]/g, '');
console.log(str0);
