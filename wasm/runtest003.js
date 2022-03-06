const {readFileSync} = require('fs')

const outputWasm = './test003.wasm';

async function run(){
    const buffer = readFileSync(outputWasm);
    const module = await WebAssembly.compile(buffer);
    const instance = await WebAssembly.instantiate(module);
    console.log(instance.exports.mul_1k_f32(3.14));
}

run();