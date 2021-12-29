const babel = require("@babel/core");
const generate = require("@babel/generator");

function generate_codes(code) {
    let result3 = babel.transformSync(code, {
        targets: "iOS 9",
        sourceMaps: true,
        presets: ["@babel/preset-env"]
    });
    let str1 = result3.code.replace(/[\n\t]/g, '');
    console.log(str1);
    //console.log(result3.map);

    let result2 = babel.transformSync(code, {
        targets: "iOS 15",
        sourceMaps: true,
        presets: ["@babel/preset-env"]
    });

    let str2 = result2.code.replace(/[\n\t]/g, '');

    ast0 = babel.transformSync(code, { ast: true });
    let str0 = ast0.code.replace(/[\n\t]/g, '');
    console.log(str0);

    console.log('------------------');
}

exports.generate_codes = generate_codes;
