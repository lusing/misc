const {readFileSync} = require('fs')

const outputWasm = './test_remu.wasm';

async function run(){
    const buffer = readFileSync(outputWasm);
    const module = await WebAssembly.compile(buffer);
    const instance = await WebAssembly.instantiate(module);
    console.log(instance.exports.rem_u_i64(1000n,256n));
}

run();
