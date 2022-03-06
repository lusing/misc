const {readFileSync, writeFileSync} = require('fs')
const wabt = require('wabt')
const path = require('path')

const inputWat = './test1.wat';
const outputWasm = './test1.wasm';

//console.log(wabt);

//console.log(wabt.parseWat)

//const wasmModule = wabt.parseWat(inputWat, readFileSync(inputWat, "utf-8"));
//const {buffer} = wasmModule.toBinary({});

//writeFileSync(outputWasm,new Buffer(buffer));

const run = async () => {
    const buffer = readFileSync(outputWasm);
    const module = await WebAssembly.compile(buffer);
    const instance = await WebAssembly.instantiate(module);
    console.log(instance.exports.consti32());
    console.log(instance.exports.consti64());
    console.log(instance.exports.const_f32());
}

run();
