const babel = require("@babel/core");
const generate = require("@babel/generator");

function generate_codes(code) {
    let result3 = babel.transformSync(code, {
        targets: "iOS 9",
        sourceMaps: true,
        presets: ["@babel/preset-env", "@babel/preset-react"]
    });
    let str1 = result3.code;//.replace(/[\n\t]/g, '');
    console.log(str1);
    //console.log(result3.map);

    let result2 = babel.transformSync(code, {
        targets: "iOS 15",
        sourceMaps: true,
        presets: ["@babel/preset-env","@babel/preset-react"]
    });

    let str2 = result2.code;//.replace(/[\n\t]/g, '');
    console.log(str2);

    ast0 = babel.transformSync(code, { ast: true, presets: ["@babel/preset-env", "@babel/preset-react"] });
    let str0 = ast0.code;//.replace(/[\n\t]/g, '');
    console.log(str0);

    console.log('------------------');
}

const codes = [
    `import React, { useState } from 'react';

    function Example() {
      const [count, setCount] = useState(0);
    
      return (
        <div>
          <p>You clicked {count} times</p>
          <button onClick={() => setCount(count + 1)}>
            Click me
          </button>
        </div>
      );
    }`
];

for (let code1 of codes) {
    generate_codes(code1);
}
