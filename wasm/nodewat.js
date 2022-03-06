const {readFile} = require('fs/promises');
const {WASI} = require('wasi');
const {argv, env} = require('process');

const wasi = new WASI({
    args: argv,
    env,
    preopens: {
        '/sandbox': './'
    }
});

async function test() {
    const importObject = {wasi_snapshot_preview1: wasi.wasiImport};

    const wasm = await WebAssembly.compile(await readFile('./test1.wasm'));
    
    const instance = await WebAssembly.instantiate(wasm, importObject);

    console.log(instance.exports.const_i32());
    console.log(instance.exports.const_i64());
    console.log(instance.exports.const_f32());
    console.log(instance.exports.add_i32());
    //wasi.start(instance);
}

test();
