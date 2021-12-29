const coder = require('./gencode');

const codes = [
    `const F = () => false;`,
    `const T = () => true;`,
    `  function _isPlaceholder(a) {
        return a != null &&
               typeof a === 'object' &&
               a['@@functional/placeholder'] === true;
      }`,
];


for (let code1 of codes) {
    coder.generate_codes(code1);
}