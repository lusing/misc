const {readFileSync} = require('fs')

const outputWasm = './test001.wasm';

async function run(){
    const buffer = readFileSync(outputWasm);
    const module = await WebAssembly.compile(buffer);
    const instance = await WebAssembly.instantiate(module);
    console.log(instance.exports.const_i32());
}

run();
