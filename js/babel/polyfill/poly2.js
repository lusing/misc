const from = require('core-js-pure/stable/array/from');
const flat = require('core-js-pure/stable/array/flat');
const Set = require('core-js-pure/stable/set');
const Promise = require('core-js-pure/stable/promise');

let polyfill2 = () => {
    from(new Set([1, 2, 3, 2, 1]));
    flat([1, [2, 3], [4, [5]]], 2);
    Promise.resolve(32).then(x => console.log(x));
    
}
polyfill2();
