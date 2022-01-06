const coder = require('./gencode');

const codes = [
    `const F = () => false;`,
    `const T = () => true;`,
    `  function _isPlaceholder(a) {
        return a != null &&
               typeof a === 'object' &&
               a['@@functional/placeholder'] === true;
      }`,
      `  function _curry1(fn) {
        return function f1(a) {
          if (arguments.length === 0 || _isPlaceholder(a)) {
            return f1;
          } else {
            return fn.apply(this, arguments);
          }
        };
      }`,
      `  function _curry2(fn) {
        return function f2(a, b) {
          switch (arguments.length) {
            case 0:
              return f2;
            case 1:
              return _isPlaceholder(a)
                ? f2
                : _curry1(function(_b) { return fn(a, _b); });
            default:
              return _isPlaceholder(a) && _isPlaceholder(b)
                ? f2
                : _isPlaceholder(a)
                  ? _curry1(function(_a) { return fn(_a, b); })
                  : _isPlaceholder(b)
                    ? _curry1(function(_b) { return fn(a, _b); })
                    : fn(a, b);
          }
        };
      }`,
      ` const add = _curry2(function add(a, b) {
        return Number(a) + Number(b);
      });`,
      `  function _concat(set1, set2) {
        set1 = set1 || [];
        set2 = set2 || [];
        let idx;
        let len1 = set1.length;
        let len2 = set2.length;
        let result = [];
    
        idx = 0;
        while (idx < len1) {
          result[result.length] = set1[idx];
          idx += 1;
        }
        idx = 0;
        while (idx < len2) {
          result[result.length] = set2[idx];
          idx += 1;
        }
        return result;
      }`,
];


for (let code1 of codes) {
    coder.generate_codes(code1);
}