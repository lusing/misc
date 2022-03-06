const {readFileSync} = require('fs')

const outputWasm = './test002.wasm';

async function run(){
    const buffer = readFileSync(outputWasm);
    const module = await WebAssembly.compile(buffer);
    const instance = await WebAssembly.instantiate(module);
    console.log(instance.exports.inc_i32(1));
}

run();