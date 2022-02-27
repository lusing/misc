const { readFileSync } = require("fs");

async function run(){
    const buffer = readFileSync("./test1.wasm");
    const module = await WebAssembly.compile(buffer);
    const instance = await WebAssembly.instantiate(module);
    console.log(instance.exports.i32_const());
    console.log(instance.exports.i32_add());
    //console.log(instance.exports.i64_add());
    console.log(instance.exports.i32_add2(1,2));
}

run();
