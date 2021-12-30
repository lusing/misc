"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const core_1 = require("@babel/core");
function add(x, y) {
    return x + y;
}
;
let f1 = (x, y) => x + y;
f1(10n, 20n);
const code = `
function add(x: number, y: number): number{
    return x+y;
}
`;
let ast0 = (0, core_1.default)(code, { ast: true, presets: ["@babel/preset-env", "@babel/preset-react"] });
let str0 = ast0.code; //.replace(/[\n\t]/g, '');
console.log(str0);
//# sourceMappingURL=index.js.map