var _fp = _interopRequireDefault(require("lodash/fp"));
==========
"use strict";

var _fp = _interopRequireDefault(require("lodash/fp"));var _function = require("./function");
==========
"use strict";

var _function = require("./function");var _collection = require("./collection");
==========
"use strict";

var _collection = require("./collection");var _conversion = require("./conversion");
==========
"use strict";

var _conversion = require("./conversion");// TODO: Move to proper files and expose
let callUnless = check => failFn => fn => (x, y) => check(x) ? failFn(y) : check(y) ? failFn(x) : fn(x, y);
==========
"use strict";

// TODO: Move to proper files and expose
var callUnless = function callUnless(check) {
  return function (failFn) {
    return function (fn) {
      return function (x, y) {
        return check(x) ? failFn(y) : check(y) ? failFn(x) : fn(x, y);
      };
    };
  };
};let callUnlessEmpty = callUnless(_fp.default.isEmpty);
==========
"use strict";

var callUnlessEmpty = callUnless(_fp.default.isEmpty);let wrapArray = x => [x];
==========
"use strict";

var wrapArray = function wrapArray(x) {
  return [x];
};let callUnlessEmptyArray = callUnlessEmpty(wrapArray);
==========
"use strict";

var callUnlessEmptyArray = callUnlessEmpty(wrapArray);let dropRight = _fp.default.dropRight(1);
==========
"use strict";

var dropRight = _fp.default.dropRight(1);let last = _fp.default.takeRight(1); // Arrays
// ------
==========
"use strict";

var last = _fp.default.takeRight(1); // Arrays
// ------// Arrays
// ------
let compactJoin = _fp.default.curry((join, x) => _fp.default.compact(x).join(join));
==========
"use strict";

// Arrays
// ------
var compactJoin = _fp.default.curry(function (join, x) {
  return _fp.default.compact(x).join(join);
});let dotJoin = compactJoin('.');
==========
"use strict";

var dotJoin = compactJoin('.');let dotJoinWith = fn => x => _fp.default.filter(fn, x).join('.');
==========
"use strict";

var dotJoinWith = function dotJoinWith(fn) {
  return function (x) {
    return _fp.default.filter(fn, x).join('.');
  };
};let repeated = _fp.default.flow(_fp.default.groupBy(e => e), _fp.default.filter(e => e.length > 1), _fp.default.flatten, _fp.default.uniq);
==========
"use strict";

var repeated = _fp.default.flow(_fp.default.groupBy(function (e) {
  return e;
}), _fp.default.filter(function (e) {
  return e.length > 1;
}), _fp.default.flatten, _fp.default.uniq);let push = _fp.default.curry((val, arr) => arr.concat([val]));
==========
"use strict";

var push = _fp.default.curry(function (val, arr) {
  return arr.concat([val]);
});let pushIn = _fp.default.curry((arr, val) => arr.concat([val]));
==========
"use strict";

var pushIn = _fp.default.curry(function (arr, val) {
  return arr.concat([val]);
});let pushOn = _fp.default.curry((arr, val) => {
  arr.push(val);
  return arr;
});
==========
"use strict";

var pushOn = _fp.default.curry(function (arr, val) {
  arr.push(val);
  return arr;
});let moveIndex = (from, to, arr) => _fp.default.flow(_fp.default.pullAt(from), (0, _collection.insertAtIndex)(to, arr[from]))(arr);
==========
"use strict";

var moveIndex = function moveIndex(from, to, arr) {
  return _fp.default.flow(_fp.default.pullAt(from), (0, _collection.insertAtIndex)(to, arr[from]))(arr);
};let overlaps = (x, y) => y[0] > x[1];
==========
"use strict";

var overlaps = function overlaps(x, y) {
  return y[0] > x[1];
};let mergeRange = (x, y) => [[x[0], _fp.default.max(x.concat(y))]];
==========
"use strict";

var mergeRange = function mergeRange(x, y) {
  return [[x[0], _fp.default.max(x.concat(y))]];
};let actuallMergeRanges = callUnlessEmptyArray((x, y) => overlaps(x, y) ? [x, y] : mergeRange(x, y));
==========
"use strict";

var actuallMergeRanges = callUnlessEmptyArray(function (x, y) {
  return overlaps(x, y) ? [x, y] : mergeRange(x, y);
});let mergeRanges = _fp.default.flow(_fp.default.sortBy([0, 1]), _fp.default.reduce((result, range) => dropRight(result).concat(actuallMergeRanges(_fp.default.flatten(last(result)), range)), [])); // [a, b...] -> a -> b
==========
"use strict";

var mergeRanges = _fp.default.flow(_fp.default.sortBy([0, 1]), _fp.default.reduce(function (result, range) {
  return dropRight(result).concat(actuallMergeRanges(_fp.default.flatten(last(result)), range));
}, [])); // [a, b...] -> a -> b// [a, b...] -> a -> b
let cycle = _fp.default.curry((a, n) => a[(a.indexOf(n) + 1) % a.length]);
==========
"use strict";

// [a, b...] -> a -> b
var cycle = _fp.default.curry(function (a, n) {
  return a[(a.indexOf(n) + 1) % a.length];
});let arrayToObject = _fp.default.curry((k, v, a) => _fp.default.flow(_fp.default.keyBy(k), _fp.default.mapValues(v))(a)); // zipObject that supports functions instead of objects
==========
"use strict";

var arrayToObject = _fp.default.curry(function (k, v, a) {
  return _fp.default.flow(_fp.default.keyBy(k), _fp.default.mapValues(v))(a);
}); // zipObject that supports functions instead of objects// zipObject that supports functions instead of objects
let zipObjectDeepWith = _fp.default.curry((x, y) => _fp.default.zipObjectDeep(x, _fp.default.isFunction(y) && _fp.default.isArray(x) ? _fp.default.times(y, x.length) : y));
==========
"use strict";

// zipObject that supports functions instead of objects
var zipObjectDeepWith = _fp.default.curry(function (x, y) {
  return _fp.default.zipObjectDeep(x, _fp.default.isFunction(y) && _fp.default.isArray(x) ? _fp.default.times(y, x.length) : y);
});let flags = zipObjectDeepWith(_fp.default, () => true);
==========
"use strict";

var flags = zipObjectDeepWith(_fp.default, function () {
  return true;
});let prefixes = list => _fp.default.range(1, list.length + 1).map(x => _fp.default.take(x, list));
==========
"use strict";

var prefixes = function prefixes(list) {
  return _fp.default.range(1, list.length + 1).map(function (x) {
    return _fp.default.take(x, list);
  });
};let encoder = separator => ({
  encode: compactJoin(separator),
  decode: _fp.default.split(separator)
});
==========
"use strict";

var encoder = function encoder(separator) {
  return {
    encode: compactJoin(separator),
    decode: _fp.default.split(separator)
  };
};let dotEncoder = encoder('.');
==========
"use strict";

var dotEncoder = encoder('.');let slashEncoder = encoder('/');
==========
"use strict";

var slashEncoder = encoder('/');let chunkBy = _fp.default.curry((f, array) => _fp.default.isEmpty(array) ? [] : _fp.default.reduce((acc, x) => f(_fp.default.last(acc), x) ? [..._fp.default.initial(acc), [..._fp.default.last(acc), x]] : [...acc, [x]], [[_fp.default.head(array)]], _fp.default.tail(array)));
==========
"use strict";

function _toConsumableArray(arr) { return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread(); }

function _nonIterableSpread() { throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }

function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }

function _iterableToArray(iter) { if (typeof Symbol !== "undefined" && iter[Symbol.iterator] != null || iter["@@iterator"] != null) return Array.from(iter); }

function _arrayWithoutHoles(arr) { if (Array.isArray(arr)) return _arrayLikeToArray(arr); }

function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }

var chunkBy = _fp.default.curry(function (f, array) {
  return _fp.default.isEmpty(array) ? [] : _fp.default.reduce(function (acc, x) {
    return f(_fp.default.last(acc), x) ? [].concat(_toConsumableArray(_fp.default.initial(acc)), [[].concat(_toConsumableArray(_fp.default.last(acc)), [x])]) : [].concat(_toConsumableArray(acc), [[x]]);
  }, [[_fp.default.head(array)]], _fp.default.tail(array));
});let toggleElementBy = _fp.default.curry((check, val, arr) => ((0, _function.callOrReturn)(check, val, arr) ? _fp.default.pull : push)(val, arr));
==========
"use strict";

var toggleElementBy = _fp.default.curry(function (check, val, arr) {
  return ((0, _function.callOrReturn)(check, val, arr) ? _fp.default.pull : push)(val, arr);
});let toggleElement = toggleElementBy(_fp.default.includes);
==========
"use strict";

var toggleElement = toggleElementBy(_fp.default.includes);let intersperse = _fp.default.curry((f, _ref) => {
  let [x0, ...xs] = _ref;
  return (0, _conversion.reduceIndexed)((acc, x, i) => i === xs.length ? [...acc, x] : [...acc, (0, _function.callOrReturn)(f, acc, i, xs), x], [x0], xs);
});
==========
"use strict";

function _toConsumableArray(arr) { return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread(); }

function _nonIterableSpread() { throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }

function _arrayWithoutHoles(arr) { if (Array.isArray(arr)) return _arrayLikeToArray(arr); }

function _toArray(arr) { return _arrayWithHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableRest(); }

function _nonIterableRest() { throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }

function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }

function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }

function _iterableToArray(iter) { if (typeof Symbol !== "undefined" && iter[Symbol.iterator] != null || iter["@@iterator"] != null) return Array.from(iter); }

function _arrayWithHoles(arr) { if (Array.isArray(arr)) return arr; }

var intersperse = _fp.default.curry(function (f, _ref) {
  var _ref2 = _toArray(_ref),
      x0 = _ref2[0],
      xs = _ref2.slice(1);

  return (0, _conversion.reduceIndexed)(function (acc, x, i) {
    return i === xs.length ? [].concat(_toConsumableArray(acc), [x]) : [].concat(_toConsumableArray(acc), [(0, _function.callOrReturn)(f, acc, i, xs), x]);
  }, [x0], xs);
});let [x0, ...xs] = _ref;
==========
"use strict";

function _toArray(arr) { return _arrayWithHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableRest(); }

function _nonIterableRest() { throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }

function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }

function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }

function _iterableToArray(iter) { if (typeof Symbol !== "undefined" && iter[Symbol.iterator] != null || iter["@@iterator"] != null) return Array.from(iter); }

function _arrayWithHoles(arr) { if (Array.isArray(arr)) return arr; }

var _ref2 = _ref,
    _ref3 = _toArray(_ref2),
    x0 = _ref3[0],
    xs = _ref3.slice(1);let replaceElementBy = _fp.default.curry((f, b, arr) => _fp.default.map(c => f(c) ? b : c, arr));
==========
"use strict";

var replaceElementBy = _fp.default.curry(function (f, b, arr) {
  return _fp.default.map(function (c) {
    return f(c) ? b : c;
  }, arr);
});let replaceElement = _fp.default.curry((a, b, arr) => replaceElementBy(_fp.default.isEqual(a), b, arr));
==========
"use strict";

var replaceElement = _fp.default.curry(function (a, b, arr) {
  return replaceElementBy(_fp.default.isEqual(a), b, arr);
});var _fp = _interopRequireDefault(require("lodash/fp"));
==========
"use strict";

var _fp = _interopRequireDefault(require("lodash/fp"));var _tree = require("./tree");
==========
"use strict";

var _tree = require("./tree");const flowMap = function () {
  return _fp.default.map(_fp.default.flow(...arguments));
};
==========
"use strict";

var flowMap = function flowMap() {
  var _fp$default;

  return _fp.default.map((_fp$default = _fp.default).flow.apply(_fp$default, arguments));
};let findApply = _fp.default.curry((f, arr) => _fp.default.iteratee(f)(_fp.default.find(f, arr))); // Algebras
// --------
// A generic map that works for plain objects and arrays
==========
"use strict";

var findApply = _fp.default.curry(function (f, arr) {
  return _fp.default.iteratee(f)(_fp.default.find(f, arr));
}); // Algebras
// --------
// A generic map that works for plain objects and arrays// Algebras
// --------
// A generic map that works for plain objects and arrays
let map = _fp.default.curry((f, x) => (_fp.default.isArray(x) ? _fp.default.map : _fp.default.mapValues).convert({
  cap: false
})(f, x)); // Map for any recursive algebraic data structure
// defaults in multidimensional arrays and recursive plain objects
==========
"use strict";

// Algebras
// --------
// A generic map that works for plain objects and arrays
var map = _fp.default.curry(function (f, x) {
  return (_fp.default.isArray(x) ? _fp.default.map : _fp.default.mapValues).convert({
    cap: false
  })(f, x);
}); // Map for any recursive algebraic data structure
// defaults in multidimensional arrays and recursive plain objects// Map for any recursive algebraic data structure
// defaults in multidimensional arrays and recursive plain objects
let deepMap = _fp.default.curry(function (fn, obj) {
  let _map = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : map;

  let is = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : _tree.isTraversable;
  return _map(e => is(e) ? deepMap(fn, fn(e), _map, is) : e, obj);
});
==========
"use strict";

// Map for any recursive algebraic data structure
// defaults in multidimensional arrays and recursive plain objects
var deepMap = _fp.default.curry(function (fn, obj) {
  var _map = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : map;

  var is = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : _tree.isTraversable;
  return _map(function (e) {
    return is(e) ? deepMap(fn, fn(e), _map, is) : e;
  }, obj);
});let _map = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : map;
==========
"use strict";

var _map = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : map;let is = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : _tree.isTraversable;
==========
"use strict";

var is = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : _tree.isTraversable;let insertAtStringIndex = (index, val, str) => str.slice(0, index) + val + str.slice(index);
==========
"use strict";

var insertAtStringIndex = function insertAtStringIndex(index, val, str) {
  return str.slice(0, index) + val + str.slice(index);
};let insertAtArrayIndex = (index, val, arr) => {
  let result = _fp.default.clone(arr);

  result.splice(index, 0, val);
  return result;
};
==========
"use strict";

var insertAtArrayIndex = function insertAtArrayIndex(index, val, arr) {
  var result = _fp.default.clone(arr);

  result.splice(index, 0, val);
  return result;
};let result = _fp.default.clone(arr);
==========
"use strict";

var result = _fp.default.clone(arr);let insertAtIndex = _fp.default.curry((index, val, collection) => _fp.default.isString(collection) ? insertAtStringIndex(index, val, collection) : insertAtArrayIndex(index, val, collection));
==========
"use strict";

var insertAtIndex = _fp.default.curry(function (index, val, collection) {
  return _fp.default.isString(collection) ? insertAtStringIndex(index, val, collection) : insertAtArrayIndex(index, val, collection);
});let compactMap = _fp.default.curry((fn, collection) => _fp.default.flow(_fp.default.map(fn), _fp.default.compact)(collection));
==========
"use strict";

var compactMap = _fp.default.curry(function (fn, collection) {
  return _fp.default.flow(_fp.default.map(fn), _fp.default.compact)(collection);
});var _fp = _interopRequireDefault(require("lodash/fp"));
==========
"use strict";

var _fp = _interopRequireDefault(require("lodash/fp"));// (fn, a, b) -> fn(a, b)
let maybeCall = function (fn) {
  for (var _len = arguments.length, args = new Array(_len > 1 ? _len - 1 : 0), _key = 1; _key < _len; _key++) {
    args[_key - 1] = arguments[_key];
  }

  return _fp.default.isFunction(fn) && fn(...args);
}; // (fn, a, b) -> fn(a, b)
==========
"use strict";

// (fn, a, b) -> fn(a, b)
var maybeCall = function maybeCall(fn) {
  for (var _len = arguments.length, args = new Array(_len > 1 ? _len - 1 : 0), _key = 1; _key < _len; _key++) {
    args[_key - 1] = arguments[_key];
  }

  return _fp.default.isFunction(fn) && fn.apply(void 0, args);
}; // (fn, a, b) -> fn(a, b)var _len = arguments.length,
    args = new Array(_len > 1 ? _len - 1 : 0),
    _key = 1;
==========
"use strict";

var _len = arguments.length,
    args = new Array(_len > 1 ? _len - 1 : 0),
    _key = 1;// (fn, a, b) -> fn(a, b)
let callOrReturn = function (fn) {
  for (var _len2 = arguments.length, args = new Array(_len2 > 1 ? _len2 - 1 : 0), _key2 = 1; _key2 < _len2; _key2++) {
    args[_key2 - 1] = arguments[_key2];
  }

  return _fp.default.isFunction(fn) ? fn(...args) : fn;
}; // (a, Monoid f) -> f[a] :: f a
==========
"use strict";

// (fn, a, b) -> fn(a, b)
var callOrReturn = function callOrReturn(fn) {
  for (var _len2 = arguments.length, args = new Array(_len2 > 1 ? _len2 - 1 : 0), _key2 = 1; _key2 < _len2; _key2++) {
    args[_key2 - 1] = arguments[_key2];
  }

  return _fp.default.isFunction(fn) ? fn.apply(void 0, args) : fn;
}; // (a, Monoid f) -> f[a] :: f avar _len2 = arguments.length,
    args = new Array(_len2 > 1 ? _len2 - 1 : 0),
    _key2 = 1;
==========
"use strict";

var _len2 = arguments.length,
    args = new Array(_len2 > 1 ? _len2 - 1 : 0),
    _key2 = 1;// (a, Monoid f) -> f[a] :: f a
let boundMethod = (method, object) => object[method].bind(object); // http://ramdajs.com/docs/#converge
==========
"use strict";

// (a, Monoid f) -> f[a] :: f a
var boundMethod = function boundMethod(method, object) {
  return object[method].bind(object);
}; // http://ramdajs.com/docs/#converge// http://ramdajs.com/docs/#converge
let converge = (converger, branches) => function () {
  return converger(_fp.default.over(branches)(...arguments));
};
==========
"use strict";

// http://ramdajs.com/docs/#converge
var converge = function converge(converger, branches) {
  return function () {
    return converger(_fp.default.over(branches).apply(void 0, arguments));
  };
};let composeApply = (f, g) => x => f(g(x))(x);
==========
"use strict";

var composeApply = function composeApply(f, g) {
  return function (x) {
    return f(g(x))(x);
  };
};let comply = composeApply; // Prettier version of `defer` the one from bluebird docs
==========
"use strict";

var comply = composeApply; // Prettier version of `defer` the one from bluebird docs// Prettier version of `defer` the one from bluebird docs
let defer = () => {
  let resolve;
  let reject;
  let promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return {
    resolve,
    reject,
    promise
  };
}; // `_.debounce` for async functions, which require consistently returning a single promise for all queued calls
==========
"use strict";

// Prettier version of `defer` the one from bluebird docs
var defer = function defer() {
  var resolve;
  var reject;
  var promise = new Promise(function (res, rej) {
    resolve = res;
    reject = rej;
  });
  return {
    resolve,
    reject,
    promise
  };
}; // `_.debounce` for async functions, which require consistently returning a single promise for all queued callslet resolve;
==========
"use strict";

var resolve;let reject;
==========
"use strict";

var reject;let promise = new Promise((res, rej) => {
  resolve = res;
  reject = rej;
});
==========
"use strict";

var promise = new Promise(function (res, rej) {
  resolve = res;
  reject = rej;
});// `_.debounce` for async functions, which require consistently returning a single promise for all queued calls
let debounceAsync = (n, f) => {
  let deferred = defer();

  let debounced = _fp.default.debounce(n, function () {
    deferred.resolve(f(...arguments));
    deferred = defer();
  });

  return function () {
    debounced(...arguments);
    return deferred.promise;
  };
};
==========
"use strict";

// `_.debounce` for async functions, which require consistently returning a single promise for all queued calls
var debounceAsync = function debounceAsync(n, f) {
  var deferred = defer();

  var debounced = _fp.default.debounce(n, function () {
    deferred.resolve(f.apply(void 0, arguments));
    deferred = defer();
  });

  return function () {
    debounced.apply(void 0, arguments);
    return deferred.promise;
  };
};let deferred = defer();
==========
"use strict";

var deferred = defer();let debounced = _fp.default.debounce(n, function () {
  deferred.resolve(f(...arguments));
  deferred = defer();
});
==========
"use strict";

var debounced = _fp.default.debounce(n, function () {
  deferred.resolve(f.apply(void 0, arguments));
  deferred = defer();
});let currier = f => function () {
  for (var _len3 = arguments.length, fns = new Array(_len3), _key3 = 0; _key3 < _len3; _key3++) {
    fns[_key3] = arguments[_key3];
  }

  return _fp.default.curryN(fns[0].length, f(...fns));
}; // (f1, f2, ...fn) -> f1Args1 -> f1Arg2 -> ...f1ArgN -> fn(f2(f1))
==========
"use strict";

var currier = function currier(f) {
  return function () {
    for (var _len3 = arguments.length, fns = new Array(_len3), _key3 = 0; _key3 < _len3; _key3++) {
      fns[_key3] = arguments[_key3];
    }

    return _fp.default.curryN(fns[0].length, f.apply(void 0, fns));
  };
}; // (f1, f2, ...fn) -> f1Args1 -> f1Arg2 -> ...f1ArgN -> fn(f2(f1))var _len3 = arguments.length,
    fns = new Array(_len3),
    _key3 = 0;
==========
"use strict";

var _len3 = arguments.length,
    fns = new Array(_len3),
    _key3 = 0;// (f1, f2, ...fn) -> f1Args1 -> f1Arg2 -> ...f1ArgN -> fn(f2(f1))
let flurry = currier(_fp.default.flow); // like _.overArgs, but on all args
==========
"use strict";

// (f1, f2, ...fn) -> f1Args1 -> f1Arg2 -> ...f1ArgN -> fn(f2(f1))
var flurry = currier(_fp.default.flow); // like _.overArgs, but on all args// like _.overArgs, but on all args
let mapArgs = _fp.default.curry((mapper, fn) => function () {
  for (var _len4 = arguments.length, x = new Array(_len4), _key4 = 0; _key4 < _len4; _key4++) {
    x[_key4] = arguments[_key4];
  }

  return fn(...x.map(mapper));
});
==========
"use strict";

function _toConsumableArray(arr) { return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread(); }

function _nonIterableSpread() { throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }

function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }

function _iterableToArray(iter) { if (typeof Symbol !== "undefined" && iter[Symbol.iterator] != null || iter["@@iterator"] != null) return Array.from(iter); }

function _arrayWithoutHoles(arr) { if (Array.isArray(arr)) return _arrayLikeToArray(arr); }

function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }

// like _.overArgs, but on all args
var mapArgs = _fp.default.curry(function (mapper, fn) {
  return function () {
    for (var _len4 = arguments.length, x = new Array(_len4), _key4 = 0; _key4 < _len4; _key4++) {
      x[_key4] = arguments[_key4];
    }

    return fn.apply(void 0, _toConsumableArray(x.map(mapper)));
  };
});var _len4 = arguments.length,
    x = new Array(_len4),
    _key4 = 0;
==========
"use strict";

var _len4 = arguments.length,
    x = new Array(_len4),
    _key4 = 0;var _fp = _interopRequireDefault(require("lodash/fp"));
==========
"use strict";

var _fp = _interopRequireDefault(require("lodash/fp"));let differentLast = (normalCase, lastCase) => (acc, i, list) => i === list.length - 1 ? _fp.default.iteratee(lastCase)(acc, i, list) : _fp.default.iteratee(normalCase)(acc, i, list);
==========
"use strict";

var differentLast = function differentLast(normalCase, lastCase) {
  return function (acc, i, list) {
    return i === list.length - 1 ? _fp.default.iteratee(lastCase)(acc, i, list) : _fp.default.iteratee(normalCase)(acc, i, list);
  };
};var _fp = _interopRequireDefault(require("lodash/fp"));
==========
"use strict";

var _fp = _interopRequireDefault(require("lodash/fp"));var _conversion = require("./conversion");
==========
"use strict";

var _conversion = require("./conversion");var _array = require("./array");
==========
"use strict";

var _array = require("./array");var _logic = require("./logic");
==========
"use strict";

var _logic = require("./logic");// Stubs
let functionLens = val => function () {
  if (!arguments.length) return val;
  val = arguments.length <= 0 ? undefined : arguments[0];
};
==========
"use strict";

// Stubs
var functionLens = function functionLens(val) {
  return function () {
    if (!arguments.length) return val;
    val = arguments.length <= 0 ? undefined : arguments[0];
  };
};let objectLens = val => ({
  get: () => val,

  set(x) {
    val = x;
  }

}); // Lens Conversion
==========
"use strict";

var objectLens = function objectLens(val) {
  return {
    get: function get() {
      return val;
    },

    set(x) {
      val = x;
    }

  };
}; // Lens Conversion// Lens Conversion
let fnToObj = fn => ({
  get: fn,
  set: fn
});
==========
"use strict";

// Lens Conversion
var fnToObj = function fnToObj(fn) {
  return {
    get: fn,
    set: fn
  };
};let objToFn = lens => function () {
  return arguments.length ? lens.set(arguments.length <= 0 ? undefined : arguments[0]) : lens.get();
}; // Lens Construction
==========
"use strict";

var objToFn = function objToFn(lens) {
  return function () {
    return arguments.length ? lens.set(arguments.length <= 0 ? undefined : arguments[0]) : lens.get();
  };
}; // Lens Construction// Lens Construction
let lensProp = (field, source) => ({
  get: () => _fp.default.get(field, source),

  //source[field],
  set(value) {
    (0, _conversion.setOn)(field, value, source); // source[field] = value
  }

}); // NOTE: This used to use mapValues; however, doing so would sometimes cause issues
// in some edge cases like trying to lens state coming from an inject function
// in the mobx library. It would inadvertently cause the inject to re-run.
// Using reduce here alleviates that issue.
==========
"use strict";

// Lens Construction
var lensProp = function lensProp(field, source) {
  return {
    get: function get() {
      return _fp.default.get(field, source);
    },

    //source[field],
    set(value) {
      (0, _conversion.setOn)(field, value, source); // source[field] = value
    }

  };
}; // NOTE: This used to use mapValues; however, doing so would sometimes cause issues
// in some edge cases like trying to lens state coming from an inject function
// in the mobx library. It would inadvertently cause the inject to re-run.
// Using reduce here alleviates that issue.// NOTE: This used to use mapValues; however, doing so would sometimes cause issues
// in some edge cases like trying to lens state coming from an inject function
// in the mobx library. It would inadvertently cause the inject to re-run.
// Using reduce here alleviates that issue.
let lensOf = object => _fp.default.reduce((res, key) => {
  res[key] = lensProp(key, object);
  return res;
}, {}, _fp.default.keys(object));
==========
"use strict";

// NOTE: This used to use mapValues; however, doing so would sometimes cause issues
// in some edge cases like trying to lens state coming from an inject function
// in the mobx library. It would inadvertently cause the inject to re-run.
// Using reduce here alleviates that issue.
var lensOf = function lensOf(object) {
  return _fp.default.reduce(function (res, key) {
    res[key] = lensProp(key, object);
    return res;
  }, {}, _fp.default.keys(object));
};let includeLens = function (value) {
  for (var _len = arguments.length, lens = new Array(_len > 1 ? _len - 1 : 0), _key = 1; _key < _len; _key++) {
    lens[_key - 1] = arguments[_key];
  }

  return {
    get: () => _fp.default.includes(value, view(...lens)),
    // Uniq is to ensure multiple calls to set(true) don't push multiple times since this is about membership of a set
    set: x => set(_fp.default.uniq((0, _array.toggleElementBy)(!x, value, view(...lens))), ...lens)
  };
}; // Lens Manipulation
//let construct = (...lens) => (lens[1] ? lensProp(...lens) : lens[0])
==========
"use strict";

var includeLens = function includeLens(value) {
  for (var _len = arguments.length, lens = new Array(_len > 1 ? _len - 1 : 0), _key = 1; _key < _len; _key++) {
    lens[_key - 1] = arguments[_key];
  }

  return {
    get: function get() {
      return _fp.default.includes(value, view.apply(void 0, lens));
    },
    // Uniq is to ensure multiple calls to set(true) don't push multiple times since this is about membership of a set
    set: function (_set) {
      function set(_x) {
        return _set.apply(this, arguments);
      }

      set.toString = function () {
        return _set.toString();
      };

      return set;
    }(function (x) {
      return set.apply(void 0, [_fp.default.uniq((0, _array.toggleElementBy)(!x, value, view.apply(void 0, lens)))].concat(lens));
    })
  };
}; // Lens Manipulation
//let construct = (...lens) => (lens[1] ? lensProp(...lens) : lens[0])var _len = arguments.length,
    lens = new Array(_len > 1 ? _len - 1 : 0),
    _key = 1;
==========
"use strict";

var _len = arguments.length,
    lens = new Array(_len > 1 ? _len - 1 : 0),
    _key = 1;// Lens Manipulation
//let construct = (...lens) => (lens[1] ? lensProp(...lens) : lens[0])
let lensPair = (get, set) => ({
  get,
  set
});
==========
"use strict";

// Lens Manipulation
//let construct = (...lens) => (lens[1] ? lensProp(...lens) : lens[0])
var lensPair = function lensPair(get, set) {
  return {
    get,
    set
  };
};let construct = function () {
  for (var _len2 = arguments.length, args = new Array(_len2), _key2 = 0; _key2 < _len2; _key2++) {
    args[_key2] = arguments[_key2];
  }

  return args[1] ? _fp.default.every(_fp.default.isFunction, args) ? lensPair(...args) : lensProp(...args) : (0, _logic.when)(_fp.default.isArray, stateLens)(args[0]);
};
==========
"use strict";

var construct = function construct() {
  for (var _len2 = arguments.length, args = new Array(_len2), _key2 = 0; _key2 < _len2; _key2++) {
    args[_key2] = arguments[_key2];
  }

  return args[1] ? _fp.default.every(_fp.default.isFunction, args) ? lensPair.apply(void 0, args) : lensProp.apply(void 0, args) : (0, _logic.when)(_fp.default.isArray, stateLens)(args[0]);
};var _len2 = arguments.length,
    args = new Array(_len2),
    _key2 = 0;
==========
"use strict";

var _len2 = arguments.length,
    args = new Array(_len2),
    _key2 = 0;let read = lens => lens.get ? lens.get() : lens();
==========
"use strict";

var read = function read(lens) {
  return lens.get ? lens.get() : lens();
};let view = function () {
  return read(construct(...arguments));
};
==========
"use strict";

var view = function view() {
  return read(construct.apply(void 0, arguments));
};let views = function () {
  for (var _len3 = arguments.length, lens = new Array(_len3), _key3 = 0; _key3 < _len3; _key3++) {
    lens[_key3] = arguments[_key3];
  }

  return () => view(...lens);
};
==========
"use strict";

var views = function views() {
  for (var _len3 = arguments.length, lens = new Array(_len3), _key3 = 0; _key3 < _len3; _key3++) {
    lens[_key3] = arguments[_key3];
  }

  return function () {
    return view.apply(void 0, lens);
  };
};var _len3 = arguments.length,
    lens = new Array(_len3),
    _key3 = 0;
==========
"use strict";

var _len3 = arguments.length,
    lens = new Array(_len3),
    _key3 = 0;let write = (val, lens) => lens.set ? lens.set(val) : lens(val);
==========
"use strict";

var write = function write(val, lens) {
  return lens.set ? lens.set(val) : lens(val);
};let set = _fp.default.curryN(2, function (val) {
  for (var _len4 = arguments.length, lens = new Array(_len4 > 1 ? _len4 - 1 : 0), _key4 = 1; _key4 < _len4; _key4++) {
    lens[_key4 - 1] = arguments[_key4];
  }

  return write(val, construct(...lens));
});
==========
"use strict";

var set = _fp.default.curryN(2, function (val) {
  for (var _len4 = arguments.length, lens = new Array(_len4 > 1 ? _len4 - 1 : 0), _key4 = 1; _key4 < _len4; _key4++) {
    lens[_key4 - 1] = arguments[_key4];
  }

  return write(val, construct.apply(void 0, lens));
});var _len4 = arguments.length,
    lens = new Array(_len4 > 1 ? _len4 - 1 : 0),
    _key4 = 1;
==========
"use strict";

var _len4 = arguments.length,
    lens = new Array(_len4 > 1 ? _len4 - 1 : 0),
    _key4 = 1;let sets = _fp.default.curryN(2, function (val) {
  for (var _len5 = arguments.length, lens = new Array(_len5 > 1 ? _len5 - 1 : 0), _key5 = 1; _key5 < _len5; _key5++) {
    lens[_key5 - 1] = arguments[_key5];
  }

  return () => set(val, ...lens);
});
==========
"use strict";

var sets = _fp.default.curryN(2, function (val) {
  for (var _len5 = arguments.length, lens = new Array(_len5 > 1 ? _len5 - 1 : 0), _key5 = 1; _key5 < _len5; _key5++) {
    lens[_key5 - 1] = arguments[_key5];
  }

  return function () {
    return set.apply(void 0, [val].concat(lens));
  };
});var _len5 = arguments.length,
    lens = new Array(_len5 > 1 ? _len5 - 1 : 0),
    _key5 = 1;
==========
"use strict";

var _len5 = arguments.length,
    lens = new Array(_len5 > 1 ? _len5 - 1 : 0),
    _key5 = 1;let setsWith = _fp.default.curry(function (f) {
  for (var _len6 = arguments.length, lens = new Array(_len6 > 1 ? _len6 - 1 : 0), _key6 = 1; _key6 < _len6; _key6++) {
    lens[_key6 - 1] = arguments[_key6];
  }

  return x => set(_fp.default.iteratee(f)(x), ...lens);
});
==========
"use strict";

var setsWith = _fp.default.curry(function (f) {
  for (var _len6 = arguments.length, lens = new Array(_len6 > 1 ? _len6 - 1 : 0), _key6 = 1; _key6 < _len6; _key6++) {
    lens[_key6 - 1] = arguments[_key6];
  }

  return function (x) {
    return set.apply(void 0, [_fp.default.iteratee(f)(x)].concat(lens));
  };
});var _len6 = arguments.length,
    lens = new Array(_len6 > 1 ? _len6 - 1 : 0),
    _key6 = 1;
==========
"use strict";

var _len6 = arguments.length,
    lens = new Array(_len6 > 1 ? _len6 - 1 : 0),
    _key6 = 1;let flip = function () {
  for (var _len7 = arguments.length, lens = new Array(_len7), _key7 = 0; _key7 < _len7; _key7++) {
    lens[_key7] = arguments[_key7];
  }

  return () => set(!view(...lens), ...lens);
};
==========
"use strict";

var flip = function flip() {
  for (var _len7 = arguments.length, lens = new Array(_len7), _key7 = 0; _key7 < _len7; _key7++) {
    lens[_key7] = arguments[_key7];
  }

  return function () {
    return set.apply(void 0, [!view.apply(void 0, lens)].concat(lens));
  };
};var _len7 = arguments.length,
    lens = new Array(_len7),
    _key7 = 0;
==========
"use strict";

var _len7 = arguments.length,
    lens = new Array(_len7),
    _key7 = 0;let on = sets(true);
==========
"use strict";

var on = sets(true);let off = sets(false); // Lens Consumption
// Map lens to dom event handlers
==========
"use strict";

var off = sets(false); // Lens Consumption
// Map lens to dom event handlers// Lens Consumption
// Map lens to dom event handlers
let binding = (value, getEventValue) => function () {
  for (var _len8 = arguments.length, lens = new Array(_len8), _key8 = 0; _key8 < _len8; _key8++) {
    lens[_key8] = arguments[_key8];
  }

  return {
    [value]: view(...lens),
    onChange: setsWith(getEventValue, ...lens)
  };
}; // Dom events have relevent fields on the `target` property of event objects
==========
"use strict";

// Lens Consumption
// Map lens to dom event handlers
var binding = function binding(value, getEventValue) {
  return function () {
    for (var _len8 = arguments.length, lens = new Array(_len8), _key8 = 0; _key8 < _len8; _key8++) {
      lens[_key8] = arguments[_key8];
    }

    return {
      [value]: view.apply(void 0, lens),
      onChange: setsWith.apply(void 0, [getEventValue].concat(lens))
    };
  };
}; // Dom events have relevent fields on the `target` property of event objectsvar _len8 = arguments.length,
    lens = new Array(_len8),
    _key8 = 0;
==========
"use strict";

var _len8 = arguments.length,
    lens = new Array(_len8),
    _key8 = 0;// Dom events have relevent fields on the `target` property of event objects
let targetBinding = field => binding(field, (0, _logic.when)(_fp.default.hasIn(`target.${field}`), _fp.default.get(`target.${field}`)));
==========
"use strict";

// Dom events have relevent fields on the `target` property of event objects
var targetBinding = function targetBinding(field) {
  return binding(field, (0, _logic.when)(_fp.default.hasIn("target.".concat(field)), _fp.default.get("target.".concat(field))));
};let domLens = {
  value: targetBinding('value'),
  checkboxValues: _fp.default.flow(includeLens, targetBinding('checked')),
  hover: function () {
    return {
      onMouseEnter: on(...arguments),
      onMouseLeave: off(...arguments)
    };
  },
  focus: function () {
    return {
      onFocus: on(...arguments),
      onBlur: off(...arguments)
    };
  },
  targetBinding,
  binding
};
==========
"use strict";

var domLens = {
  value: targetBinding('value'),
  checkboxValues: _fp.default.flow(includeLens, targetBinding('checked')),
  hover: function hover() {
    return {
      onMouseEnter: on.apply(void 0, arguments),
      onMouseLeave: off.apply(void 0, arguments)
    };
  },
  focus: function focus() {
    return {
      onFocus: on.apply(void 0, arguments),
      onBlur: off.apply(void 0, arguments)
    };
  },
  targetBinding,
  binding
};let stateLens = _ref => {
  let [value, set] = _ref;
  return {
    get: () => value,
    set
  };
};
==========
"use strict";

function _slicedToArray(arr, i) { return _arrayWithHoles(arr) || _iterableToArrayLimit(arr, i) || _unsupportedIterableToArray(arr, i) || _nonIterableRest(); }

function _nonIterableRest() { throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }

function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }

function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }

function _iterableToArrayLimit(arr, i) { var _i = arr == null ? null : typeof Symbol !== "undefined" && arr[Symbol.iterator] || arr["@@iterator"]; if (_i == null) return; var _arr = []; var _n = true; var _d = false; var _s, _e; try { for (_i = _i.call(arr); !(_n = (_s = _i.next()).done); _n = true) { _arr.push(_s.value); if (i && _arr.length === i) break; } } catch (err) { _d = true; _e = err; } finally { try { if (!_n && _i["return"] != null) _i["return"](); } finally { if (_d) throw _e; } } return _arr; }

function _arrayWithHoles(arr) { if (Array.isArray(arr)) return arr; }

var stateLens = function stateLens(_ref) {
  var _ref2 = _slicedToArray(_ref, 2),
      value = _ref2[0],
      set = _ref2[1];

  return {
    get: function get() {
      return value;
    },
    set
  };
};let [value, set] = _ref;
==========
"use strict";

function _slicedToArray(arr, i) { return _arrayWithHoles(arr) || _iterableToArrayLimit(arr, i) || _unsupportedIterableToArray(arr, i) || _nonIterableRest(); }

function _nonIterableRest() { throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }

function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }

function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }

function _iterableToArrayLimit(arr, i) { var _i = arr == null ? null : typeof Symbol !== "undefined" && arr[Symbol.iterator] || arr["@@iterator"]; if (_i == null) return; var _arr = []; var _n = true; var _d = false; var _s, _e; try { for (_i = _i.call(arr); !(_n = (_s = _i.next()).done); _n = true) { _arr.push(_s.value); if (i && _arr.length === i) break; } } catch (err) { _d = true; _e = err; } finally { try { if (!_n && _i["return"] != null) _i["return"](); } finally { if (_d) throw _e; } } return _arr; }

function _arrayWithHoles(arr) { if (Array.isArray(arr)) return arr; }

var _ref2 = _ref,
    _ref3 = _slicedToArray(_ref2, 2),
    value = _ref3[0],
    set = _ref3[1];var _fp = _interopRequireDefault(require("lodash/fp"));
==========
"use strict";

var _fp = _interopRequireDefault(require("lodash/fp"));var _array = require("./array");
==========
"use strict";

var _array = require("./array");var _logic = require("./logic");
==========
"use strict";

var _logic = require("./logic");var _lang = require("./lang");
==========
"use strict";

var _lang = require("./lang");var _conversion = require("./conversion");
==========
"use strict";

var _conversion = require("./conversion");var _collection = require("./collection");
==========
"use strict";

var _collection = require("./collection");var _aspect = require("./aspect");
==========
"use strict";

var _aspect = require("./aspect");var _function = require("./function");
==========
"use strict";

var _function = require("./function");const noCap = _fp.default.convert({
  cap: false
}); // (k, v) -> {k: v}
==========
"use strict";

var noCap = _fp.default.convert({
  cap: false
}); // (k, v) -> {k: v}// (k, v) -> {k: v}
const singleObject = _fp.default.curry((key, value) => ({
  [key]: value
}));
==========
"use strict";

// (k, v) -> {k: v}
var singleObject = _fp.default.curry(function (key, value) {
  return {
    [key]: value
  };
});const singleObjectR = _fp.default.flip(singleObject); // Formerly objToObjArr
// ({a, b}) -> [{a}, {b}]
==========
"use strict";

var singleObjectR = _fp.default.flip(singleObject); // Formerly objToObjArr
// ({a, b}) -> [{a}, {b}]// Formerly objToObjArr
// ({a, b}) -> [{a}, {b}]
const chunkObject = value => _fp.default.isArray(value) ? value : _fp.default.map(_fp.default.spread(singleObject), _fp.default.toPairs(value)); // Remove properties with falsey values: ({ a: 1, b: null, c: false}) -> {a:1}
==========
"use strict";

// Formerly objToObjArr
// ({a, b}) -> [{a}, {b}]
var chunkObject = function chunkObject(value) {
  return _fp.default.isArray(value) ? value : _fp.default.map(_fp.default.spread(singleObject), _fp.default.toPairs(value));
}; // Remove properties with falsey values: ({ a: 1, b: null, c: false}) -> {a:1}// Remove properties with falsey values: ({ a: 1, b: null, c: false}) -> {a:1}
const compactObject = _fp.default.pickBy(_fp.default.identity);
==========
"use strict";

// Remove properties with falsey values: ({ a: 1, b: null, c: false}) -> {a:1}
var compactObject = _fp.default.pickBy(_fp.default.identity);const isEmptyObject = _fp.default.isEqual({});
==========
"use strict";

var isEmptyObject = _fp.default.isEqual({});const isNotEmptyObject = _fp.default.negate(isEmptyObject); // { a:1, b:{}, c:2 } -> { a:1, c:2 }
==========
"use strict";

var isNotEmptyObject = _fp.default.negate(isEmptyObject); // { a:1, b:{}, c:2 } -> { a:1, c:2 }// { a:1, b:{}, c:2 } -> { a:1, c:2 }
const stripEmptyObjects = _fp.default.pickBy(isNotEmptyObject); // const crazyBS = (f, g) => (a, b) => f(a)(g(b))
// { a: { b: 1, c: 2 } }, [ 'b' ] -> { a: { b: 1 } }
==========
"use strict";

// { a:1, b:{}, c:2 } -> { a:1, c:2 }
var stripEmptyObjects = _fp.default.pickBy(isNotEmptyObject); // const crazyBS = (f, g) => (a, b) => f(a)(g(b))
// { a: { b: 1, c: 2 } }, [ 'b' ] -> { a: { b: 1 } }// const crazyBS = (f, g) => (a, b) => f(a)(g(b))
// { a: { b: 1, c: 2 } }, [ 'b' ] -> { a: { b: 1 } }
const pickInto = (map, source) => _fp.default.mapValues((0, _conversion.pickIn)(source), map);
==========
"use strict";

// const crazyBS = (f, g) => (a, b) => f(a)(g(b))
// { a: { b: 1, c: 2 } }, [ 'b' ] -> { a: { b: 1 } }
var pickInto = function pickInto(map, source) {
  return _fp.default.mapValues((0, _conversion.pickIn)(source), map);
};const renameProperty = _fp.default.curry((from, to, target) => _fp.default.has(from, target) ? _fp.default.flow(x => _fp.default.set(to, _fp.default.get(from, x), x), _fp.default.unset(from))(target) : target); // { x:['a','b'], y:1 } -> [{ x:'a', y:1 }, { x:'b', y:1 }] just like mongo's `$unwind`
==========
"use strict";

var renameProperty = _fp.default.curry(function (from, to, target) {
  return _fp.default.has(from, target) ? _fp.default.flow(function (x) {
    return _fp.default.set(to, _fp.default.get(from, x), x);
  }, _fp.default.unset(from))(target) : target;
}); // { x:['a','b'], y:1 } -> [{ x:'a', y:1 }, { x:'b', y:1 }] just like mongo's `$unwind`// { x:['a','b'], y:1 } -> [{ x:'a', y:1 }, { x:'b', y:1 }] just like mongo's `$unwind`
const unwind = _fp.default.curry((prop, x) => (0, _logic.ifElse)(_fp.default.isArray, _fp.default.map(y => _fp.default.set(prop, y, x)), _fp.default.stubArray, _fp.default.get(prop, x))); // this one's _actually_ just like mongo's `$unwind`
==========
"use strict";

// { x:['a','b'], y:1 } -> [{ x:'a', y:1 }, { x:'b', y:1 }] just like mongo's `$unwind`
var unwind = _fp.default.curry(function (prop, x) {
  return (0, _logic.ifElse)(_fp.default.isArray, _fp.default.map(function (y) {
    return _fp.default.set(prop, y, x);
  }), _fp.default.stubArray, _fp.default.get(prop, x));
}); // this one's _actually_ just like mongo's `$unwind`// this one's _actually_ just like mongo's `$unwind`
const unwindArray = _fp.default.curry((prop, xs) => _fp.default.flatMap(unwind(prop))(xs));
==========
"use strict";

// this one's _actually_ just like mongo's `$unwind`
var unwindArray = _fp.default.curry(function (prop, xs) {
  return _fp.default.flatMap(unwind(prop))(xs);
});const isFlatObject = (0, _logic.overNone)([_fp.default.isPlainObject, _fp.default.isArray]); // { a: { b: { c: 1 } } } => { 'a.b.c' : 1 }
==========
"use strict";

var isFlatObject = (0, _logic.overNone)([_fp.default.isPlainObject, _fp.default.isArray]); // { a: { b: { c: 1 } } } => { 'a.b.c' : 1 }// { a: { b: { c: 1 } } } => { 'a.b.c' : 1 }
const flattenObject = (input, paths) => (0, _conversion.reduceIndexed)((output, value, key) => _fp.default.merge(output, (isFlatObject(value) ? singleObjectR : flattenObject)(value, (0, _array.dotJoinWith)(_lang.isNotNil)([paths, key]))), {}, input); // { 'a.b.c' : 1 } => { a: { b: { c: 1 } } }
==========
"use strict";

// { a: { b: { c: 1 } } } => { 'a.b.c' : 1 }
var flattenObject = function flattenObject(input, paths) {
  return (0, _conversion.reduceIndexed)(function (output, value, key) {
    return _fp.default.merge(output, (isFlatObject(value) ? singleObjectR : flattenObject)(value, (0, _array.dotJoinWith)(_lang.isNotNil)([paths, key])));
  }, {}, input);
}; // { 'a.b.c' : 1 } => { a: { b: { c: 1 } } }// { 'a.b.c' : 1 } => { a: { b: { c: 1 } } }
const unflattenObject = x => _fp.default.zipObjectDeep(_fp.default.keys(x), _fp.default.values(x)); // Returns true if object keys are only elements from signature list (but does not require all signature keys to be present)
==========
"use strict";

// { 'a.b.c' : 1 } => { a: { b: { c: 1 } } }
var unflattenObject = function unflattenObject(x) {
  return _fp.default.zipObjectDeep(_fp.default.keys(x), _fp.default.values(x));
}; // Returns true if object keys are only elements from signature list (but does not require all signature keys to be present)// Returns true if object keys are only elements from signature list (but does not require all signature keys to be present)
const matchesSignature = _fp.default.curry((signature, value) => _fp.default.isObject(value) && !_fp.default.difference(_fp.default.keys(value), signature).length); // `_.matches` that returns true if one or more of the conditions match instead of all
==========
"use strict";

// Returns true if object keys are only elements from signature list (but does not require all signature keys to be present)
var matchesSignature = _fp.default.curry(function (signature, value) {
  return _fp.default.isObject(value) && !_fp.default.difference(_fp.default.keys(value), signature).length;
}); // `_.matches` that returns true if one or more of the conditions match instead of all// `_.matches` that returns true if one or more of the conditions match instead of all
const matchesSome = _fp.default.flow(chunkObject, _fp.default.map(_fp.default.matches), _fp.default.overSome); // Checks if a property deep in a given item equals to a given value
==========
"use strict";

// `_.matches` that returns true if one or more of the conditions match instead of all
var matchesSome = _fp.default.flow(chunkObject, _fp.default.map(_fp.default.matches), _fp.default.overSome); // Checks if a property deep in a given item equals to a given value// Checks if a property deep in a given item equals to a given value
const compareDeep = _fp.default.curry((path, item, value) => _fp.default.get(path, item) === value); //Depreacted in favor of _.update version from lodash
==========
"use strict";

// Checks if a property deep in a given item equals to a given value
var compareDeep = _fp.default.curry(function (path, item, value) {
  return _fp.default.get(path, item) === value;
}); //Depreacted in favor of _.update version from lodash//Depreacted in favor of _.update version from lodash
const mapProp = _aspect.aspects.deprecate('mapProp', '1.46.0', '_.update')(noCap.update); // `_.get` that returns the target object if lookup fails
==========
"use strict";

//Depreacted in favor of _.update version from lodash
var mapProp = _aspect.aspects.deprecate('mapProp', '1.46.0', '_.update')(noCap.update); // `_.get` that returns the target object if lookup fails// `_.get` that returns the target object if lookup fails
let getOrReturn = _fp.default.curry((prop, x) => _fp.default.getOr(x, prop, x)); // `_.get` that returns the prop if lookup fails
==========
"use strict";

// `_.get` that returns the target object if lookup fails
var getOrReturn = _fp.default.curry(function (prop, x) {
  return _fp.default.getOr(x, prop, x);
}); // `_.get` that returns the prop if lookup fails// `_.get` that returns the prop if lookup fails
let alias = _fp.default.curry((prop, x) => _fp.default.getOr(prop, prop, x)); // flipped alias
==========
"use strict";

// `_.get` that returns the prop if lookup fails
var alias = _fp.default.curry(function (prop, x) {
  return _fp.default.getOr(prop, prop, x);
}); // flipped alias// flipped alias
let aliasIn = _fp.default.curry((x, prop) => _fp.default.getOr(prop, prop, x)); // A `_.get` that takes an array of paths and returns the value at the first path that matches
==========
"use strict";

// flipped alias
var aliasIn = _fp.default.curry(function (x, prop) {
  return _fp.default.getOr(prop, prop, x);
}); // A `_.get` that takes an array of paths and returns the value at the first path that matches// A `_.get` that takes an array of paths and returns the value at the first path that matches
let cascade = _fp.default.curryN(2, (paths, obj, defaultValue) => _fp.default.flow((0, _collection.findApply)(x => x && _fp.default.iteratee(x)(obj)), _fp.default.defaultTo(defaultValue))(paths)); // Flipped cascade
==========
"use strict";

// A `_.get` that takes an array of paths and returns the value at the first path that matches
var cascade = _fp.default.curryN(2, function (paths, obj, defaultValue) {
  return _fp.default.flow((0, _collection.findApply)(function (x) {
    return x && _fp.default.iteratee(x)(obj);
  }), _fp.default.defaultTo(defaultValue))(paths);
}); // Flipped cascade// Flipped cascade
let cascadeIn = _fp.default.curryN(2, (obj, paths, defaultValue) => cascade(paths, obj, defaultValue)); // A `_.get` that takes an array of paths and returns the first path that matched
==========
"use strict";

// Flipped cascade
var cascadeIn = _fp.default.curryN(2, function (obj, paths, defaultValue) {
  return cascade(paths, obj, defaultValue);
}); // A `_.get` that takes an array of paths and returns the first path that matched// A `_.get` that takes an array of paths and returns the first path that matched
let cascadeKey = _fp.default.curry((paths, obj) => _fp.default.find((0, _conversion.getIn)(obj), paths)); // A `_.get` that takes an array of paths and returns the first path that exists
==========
"use strict";

// A `_.get` that takes an array of paths and returns the first path that matched
var cascadeKey = _fp.default.curry(function (paths, obj) {
  return _fp.default.find((0, _conversion.getIn)(obj), paths);
}); // A `_.get` that takes an array of paths and returns the first path that exists// A `_.get` that takes an array of paths and returns the first path that exists
let cascadePropKey = _fp.default.curry((paths, obj) => _fp.default.find((0, _conversion.hasIn)(obj), paths)); // A `_.get` that takes an array of paths and returns the first value that has an existing path
==========
"use strict";

// A `_.get` that takes an array of paths and returns the first path that exists
var cascadePropKey = _fp.default.curry(function (paths, obj) {
  return _fp.default.find((0, _conversion.hasIn)(obj), paths);
}); // A `_.get` that takes an array of paths and returns the first value that has an existing path// A `_.get` that takes an array of paths and returns the first value that has an existing path
let cascadeProp = _fp.default.curry((paths, obj) => _fp.default.get(cascadePropKey(paths, obj), obj));
==========
"use strict";

// A `_.get` that takes an array of paths and returns the first value that has an existing path
var cascadeProp = _fp.default.curry(function (paths, obj) {
  return _fp.default.get(cascadePropKey(paths, obj), obj);
});let unkeyBy = _fp.default.curry((keyName, obj) => (0, _conversion.mapIndexed)((val, key) => _fp.default.extend(val, {
  [keyName || key]: key
}))(obj));
==========
"use strict";

var unkeyBy = _fp.default.curry(function (keyName, obj) {
  return (0, _conversion.mapIndexed)(function (val, key) {
    return _fp.default.extend(val, {
      [keyName || key]: key
    });
  })(obj);
});let simpleDiff = (original, deltas) => {
  let o = flattenObject(original);
  return _fp.default.flow(flattenObject, (0, _conversion.mapValuesIndexed)((to, field) => ({
    from: o[field],
    to
  })), _fp.default.omitBy(x => _fp.default.isEqual(x.from, x.to)))(deltas);
};
==========
"use strict";

var simpleDiff = function simpleDiff(original, deltas) {
  var o = flattenObject(original);
  return _fp.default.flow(flattenObject, (0, _conversion.mapValuesIndexed)(function (to, field) {
    return {
      from: o[field],
      to
    };
  }), _fp.default.omitBy(function (x) {
    return _fp.default.isEqual(x.from, x.to);
  }))(deltas);
};let o = flattenObject(original);
==========
"use strict";

var o = flattenObject(original);let simpleDiffArray = _fp.default.flow(simpleDiff, unkeyBy('field'));
==========
"use strict";

var simpleDiffArray = _fp.default.flow(simpleDiff, unkeyBy('field'));let diff = (original, deltas) => {
  let o = flattenObject(original);
  let d = flattenObject(deltas);
  return _fp.default.flow((0, _conversion.mapValuesIndexed)((_, field) => ({
    from: o[field],
    to: d[field]
  })), _fp.default.omitBy(x => _fp.default.isEqual(x.from, x.to)))(_fp.default.merge(o, d));
};
==========
"use strict";

var diff = function diff(original, deltas) {
  var o = flattenObject(original);
  var d = flattenObject(deltas);
  return _fp.default.flow((0, _conversion.mapValuesIndexed)(function (_, field) {
    return {
      from: o[field],
      to: d[field]
    };
  }), _fp.default.omitBy(function (x) {
    return _fp.default.isEqual(x.from, x.to);
  }))(_fp.default.merge(o, d));
};let o = flattenObject(original);
==========
"use strict";

var o = flattenObject(original);let d = flattenObject(deltas);
==========
"use strict";

var d = flattenObject(deltas);let diffArray = _fp.default.flow(diff, unkeyBy('field')); // A `_.pick` that mutates the object
==========
"use strict";

var diffArray = _fp.default.flow(diff, unkeyBy('field')); // A `_.pick` that mutates the object// A `_.pick` that mutates the object
let pickOn = function () {
  let paths = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : [];
  let obj = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};
  return _fp.default.flow(_fp.default.keys, _fp.default.map(key => {
    if (!_fp.default.includes(key, paths)) {
      delete obj[key];
    }
  }))(obj);
};
==========
"use strict";

// A `_.pick` that mutates the object
var pickOn = function pickOn() {
  var paths = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : [];
  var obj = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};
  return _fp.default.flow(_fp.default.keys, _fp.default.map(function (key) {
    if (!_fp.default.includes(key, paths)) {
      delete obj[key];
    }
  }))(obj);
};let paths = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : [];
==========
"use strict";

var paths = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : [];let obj = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};
==========
"use strict";

var obj = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};let mergeArrays = (objValue, srcValue) => _fp.default.isArray(objValue) ? objValue.concat(srcValue) : undefined; // Straight from the lodash docs
==========
"use strict";

var mergeArrays = function mergeArrays(objValue, srcValue) {
  return _fp.default.isArray(objValue) ? objValue.concat(srcValue) : undefined;
}; // Straight from the lodash docs// Straight from the lodash docs
let mergeAllArrays = _fp.default.mergeAllWith(mergeArrays); // { a: [x, y, z], b: [x] } -> { x: [a, b], y: [a], z: [a] }
==========
"use strict";

// Straight from the lodash docs
var mergeAllArrays = _fp.default.mergeAllWith(mergeArrays); // { a: [x, y, z], b: [x] } -> { x: [a, b], y: [a], z: [a] }// { a: [x, y, z], b: [x] } -> { x: [a, b], y: [a], z: [a] }
let invertByArray = _fp.default.flow((0, _conversion.mapIndexed)((arr, key) => (0, _array.zipObjectDeepWith)(arr, () => [key])), mergeAllArrays); // key -> { a: { x: 1 }, b: { y: 2 } } -> { a: { x: 1, key: 'a' }, b: { y: 2, key: 'b' } }
==========
"use strict";

// { a: [x, y, z], b: [x] } -> { x: [a, b], y: [a], z: [a] }
var invertByArray = _fp.default.flow((0, _conversion.mapIndexed)(function (arr, key) {
  return (0, _array.zipObjectDeepWith)(arr, function () {
    return [key];
  });
}), mergeAllArrays); // key -> { a: { x: 1 }, b: { y: 2 } } -> { a: { x: 1, key: 'a' }, b: { y: 2, key: 'b' } }// key -> { a: { x: 1 }, b: { y: 2 } } -> { a: { x: 1, key: 'a' }, b: { y: 2, key: 'b' } }
const stampKey = _fp.default.curry((key, x) => (0, _conversion.mapValuesIndexed)((val, k) => ({ ...val,
  [key]: k
}), x));
==========
"use strict";

function ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); enumerableOnly && (symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; })), keys.push.apply(keys, symbols); } return keys; }

function _objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = null != arguments[i] ? arguments[i] : {}; i % 2 ? ownKeys(Object(source), !0).forEach(function (key) { _defineProperty(target, key, source[key]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } return target; }

function _defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }

// key -> { a: { x: 1 }, b: { y: 2 } } -> { a: { x: 1, key: 'a' }, b: { y: 2, key: 'b' } }
var stampKey = _fp.default.curry(function (key, x) {
  return (0, _conversion.mapValuesIndexed)(function (val, k) {
    return _objectSpread(_objectSpread({}, val), {}, {
      [key]: k
    });
  }, x);
});let omitNil = x => _fp.default.omitBy(_fp.default.isNil, x);
==========
"use strict";

var omitNil = function omitNil(x) {
  return _fp.default.omitBy(_fp.default.isNil, x);
};let omitNull = x => _fp.default.omitBy(_fp.default.isNull, x);
==========
"use strict";

var omitNull = function omitNull(x) {
  return _fp.default.omitBy(_fp.default.isNull, x);
};let omitBlank = x => _fp.default.omitBy(_lang.isBlank, x);
==========
"use strict";

var omitBlank = function omitBlank(x) {
  return _fp.default.omitBy(_lang.isBlank, x);
};let omitEmpty = x => _fp.default.omitBy(_fp.default.isEmpty, x); // ([f, g]) -> (x, y) -> {...f(x, y), ...g(x, y)}
==========
"use strict";

var omitEmpty = function omitEmpty(x) {
  return _fp.default.omitBy(_fp.default.isEmpty, x);
}; // ([f, g]) -> (x, y) -> {...f(x, y), ...g(x, y)}// ([f, g]) -> (x, y) -> {...f(x, y), ...g(x, y)}
let mergeOverAll = _fp.default.curryN(2, function (fns) {
  for (var _len = arguments.length, x = new Array(_len > 1 ? _len - 1 : 0), _key = 1; _key < _len; _key++) {
    x[_key - 1] = arguments[_key];
  }

  return _fp.default.flow(_fp.default.over(fns), _fp.default.mergeAll)(...x);
}); // customizer -> ([f, g]) -> (x, y) -> {...f(x, y), ...g(x, y)}
==========
"use strict";

// ([f, g]) -> (x, y) -> {...f(x, y), ...g(x, y)}
var mergeOverAll = _fp.default.curryN(2, function (fns) {
  for (var _len = arguments.length, x = new Array(_len > 1 ? _len - 1 : 0), _key = 1; _key < _len; _key++) {
    x[_key - 1] = arguments[_key];
  }

  return _fp.default.flow(_fp.default.over(fns), _fp.default.mergeAll).apply(void 0, x);
}); // customizer -> ([f, g]) -> (x, y) -> {...f(x, y), ...g(x, y)}var _len = arguments.length,
    x = new Array(_len > 1 ? _len - 1 : 0),
    _key = 1;
==========
"use strict";

var _len = arguments.length,
    x = new Array(_len > 1 ? _len - 1 : 0),
    _key = 1;// customizer -> ([f, g]) -> (x, y) -> {...f(x, y), ...g(x, y)}
let mergeOverAllWith = _fp.default.curryN(3, function (customizer, fns) {
  for (var _len2 = arguments.length, x = new Array(_len2 > 2 ? _len2 - 2 : 0), _key2 = 2; _key2 < _len2; _key2++) {
    x[_key2 - 2] = arguments[_key2];
  }

  return _fp.default.flow(_fp.default.over(fns), _fp.default.mergeAllWith(customizer))(...x);
}); // ([f, g]) -> (x, y) -> {...f(x, y), ...g(x, y)}
==========
"use strict";

// customizer -> ([f, g]) -> (x, y) -> {...f(x, y), ...g(x, y)}
var mergeOverAllWith = _fp.default.curryN(3, function (customizer, fns) {
  for (var _len2 = arguments.length, x = new Array(_len2 > 2 ? _len2 - 2 : 0), _key2 = 2; _key2 < _len2; _key2++) {
    x[_key2 - 2] = arguments[_key2];
  }

  return _fp.default.flow(_fp.default.over(fns), _fp.default.mergeAllWith(customizer)).apply(void 0, x);
}); // ([f, g]) -> (x, y) -> {...f(x, y), ...g(x, y)}var _len2 = arguments.length,
    x = new Array(_len2 > 2 ? _len2 - 2 : 0),
    _key2 = 2;
==========
"use strict";

var _len2 = arguments.length,
    x = new Array(_len2 > 2 ? _len2 - 2 : 0),
    _key2 = 2;// ([f, g]) -> (x, y) -> {...f(x, y), ...g(x, y)}
let mergeOverAllArrays = mergeOverAllWith(mergeArrays); // (x -> y) -> k -> {k: x} -> y
==========
"use strict";

// ([f, g]) -> (x, y) -> {...f(x, y), ...g(x, y)}
var mergeOverAllArrays = mergeOverAllWith(mergeArrays); // (x -> y) -> k -> {k: x} -> y// (x -> y) -> k -> {k: x} -> y
let getWith = _fp.default.curry((customizer, path, object) => customizer(_fp.default.get(path, object))); // ({a} -> {b}) -> {a} -> {a, b}
==========
"use strict";

// (x -> y) -> k -> {k: x} -> y
var getWith = _fp.default.curry(function (customizer, path, object) {
  return customizer(_fp.default.get(path, object));
}); // ({a} -> {b}) -> {a} -> {a, b}// ({a} -> {b}) -> {a} -> {a, b}
let expandObject = _fp.default.curry((transform, obj) => ({ ...obj,
  ...transform(obj)
})); // k -> (a -> {b}) -> {k: a} -> {a, b}
==========
"use strict";

function ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); enumerableOnly && (symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; })), keys.push.apply(keys, symbols); } return keys; }

function _objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = null != arguments[i] ? arguments[i] : {}; i % 2 ? ownKeys(Object(source), !0).forEach(function (key) { _defineProperty(target, key, source[key]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } return target; }

function _defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }

// ({a} -> {b}) -> {a} -> {a, b}
var expandObject = _fp.default.curry(function (transform, obj) {
  return _objectSpread(_objectSpread({}, obj), transform(obj));
}); // k -> (a -> {b}) -> {k: a} -> {a, b}// k -> (a -> {b}) -> {k: a} -> {a, b}
let expandObjectBy = _fp.default.curry((key, fn, obj) => expandObject(getWith(fn, key))(obj));
==========
"use strict";

// k -> (a -> {b}) -> {k: a} -> {a, b}
var expandObjectBy = _fp.default.curry(function (key, fn, obj) {
  return expandObject(getWith(fn, key))(obj);
});let commonKeys = _fp.default.curryN(2, (0, _function.mapArgs)(_fp.default.keys, _fp.default.intersection));
==========
"use strict";

var commonKeys = _fp.default.curryN(2, (0, _function.mapArgs)(_fp.default.keys, _fp.default.intersection));let findKeyIndexed = _fp.default.findKey.convert({
  cap: false
});
==========
"use strict";

var findKeyIndexed = _fp.default.findKey.convert({
  cap: false
});let firstCommonKey = _fp.default.curry((x, y) => findKeyIndexed((val, key) => _fp.default.has(key, x), y));
==========
"use strict";

var firstCommonKey = _fp.default.curry(function (x, y) {
  return findKeyIndexed(function (val, key) {
    return _fp.default.has(key, x);
  }, y);
});var _collection = require("./collection");
==========
"use strict";

var _collection = require("./collection");var _fp = _interopRequireDefault(require("lodash/fp"));
==========
"use strict";

var _fp = _interopRequireDefault(require("lodash/fp"));var _logic = require("./logic");
==========
"use strict";

var _logic = require("./logic");var _array = require("./array");
==========
"use strict";

var _array = require("./array");var _iterators = require("./iterators");
==========
"use strict";

var _iterators = require("./iterators");const wrap = (pre, post, content) => (pre || '') + content + (post || pre || '');
==========
"use strict";

var wrap = function wrap(pre, post, content) {
  return (pre || '') + content + (post || pre || '');
};const quote = _fp.default.partial(wrap, ['"', '"']);
==========
"use strict";

var quote = _fp.default.partial(wrap, ['"', '"']);const parens = _fp.default.partial(wrap, ['(', ')']);
==========
"use strict";

var parens = _fp.default.partial(wrap, ['(', ')']);const concatStrings = _fp.default.flow(_fp.default.compact, _fp.default.map(_fp.default.trim), _fp.default.join(' '));
==========
"use strict";

var concatStrings = _fp.default.flow(_fp.default.compact, _fp.default.map(_fp.default.trim), _fp.default.join(' '));const trimStrings = (0, _collection.map)((0, _logic.when)(_fp.default.isString, _fp.default.trim)); // _.startCase does the trick, deprecate it!
==========
"use strict";

var trimStrings = (0, _collection.map)((0, _logic.when)(_fp.default.isString, _fp.default.trim)); // _.startCase does the trick, deprecate it!// _.startCase does the trick, deprecate it!
let autoLabel = _fp.default.startCase;
==========
"use strict";

// _.startCase does the trick, deprecate it!
var autoLabel = _fp.default.startCase;let autoLabelOption = a => ({
  value: (0, _logic.when)(_fp.default.isUndefined, a)(a.value),
  label: a.label || autoLabel((0, _logic.when)(_fp.default.isUndefined, a)(a.value))
});
==========
"use strict";

var autoLabelOption = function autoLabelOption(a) {
  return {
    value: (0, _logic.when)(_fp.default.isUndefined, a)(a.value),
    label: a.label || autoLabel((0, _logic.when)(_fp.default.isUndefined, a)(a.value))
  };
};let autoLabelOptions = _fp.default.map(autoLabelOption);
==========
"use strict";

var autoLabelOptions = _fp.default.map(autoLabelOption);let toSentenceWith = _fp.default.curry((separator, lastSeparator, array) => _fp.default.flow((0, _array.intersperse)((0, _iterators.differentLast)(() => separator, () => lastSeparator)), _fp.default.join(''))(array));
==========
"use strict";

var toSentenceWith = _fp.default.curry(function (separator, lastSeparator, array) {
  return _fp.default.flow((0, _array.intersperse)((0, _iterators.differentLast)(function () {
    return separator;
  }, function () {
    return lastSeparator;
  })), _fp.default.join(''))(array);
});let toSentence = toSentenceWith(', ', ' and '); // ((array -> object), array) -> string -> string
==========
"use strict";

var toSentence = toSentenceWith(', ', ' and '); // ((array -> object), array) -> string -> string// ((array -> object), array) -> string -> string
let uniqueStringWith = _fp.default.curry((cachizer, initialKeys) => {
  let f = x => {
    let result = x;

    while (cache[result]) {
      result = x + cache[x];
      cache[x] += 1;
    }

    cache[result] = (cache[result] || 0) + 1;
    return result;
  };

  let cache = cachizer(initialKeys);
  f.cache = cache;

  f.clear = () => {
    cache = {};
    f.cache = cache;
  };

  return f;
});
==========
"use strict";

// ((array -> object), array) -> string -> string
var uniqueStringWith = _fp.default.curry(function (cachizer, initialKeys) {
  var f = function f(x) {
    var result = x;

    while (cache[result]) {
      result = x + cache[x];
      cache[x] += 1;
    }

    cache[result] = (cache[result] || 0) + 1;
    return result;
  };

  var cache = cachizer(initialKeys);
  f.cache = cache;

  f.clear = function () {
    cache = {};
    f.cache = cache;
  };

  return f;
});let f = x => {
  let result = x;

  while (cache[result]) {
    result = x + cache[x];
    cache[x] += 1;
  }

  cache[result] = (cache[result] || 0) + 1;
  return result;
};
==========
"use strict";

var f = function f(x) {
  var result = x;

  while (cache[result]) {
    result = x + cache[x];
    cache[x] += 1;
  }

  cache[result] = (cache[result] || 0) + 1;
  return result;
};let result = x;
==========
"use strict";

var result = x;let cache = cachizer(initialKeys);
==========
"use strict";

var cache = cachizer(initialKeys);let uniqueString = function () {
  let arr = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : [];
  return uniqueStringWith(_fp.default.countBy(_fp.default.identity), arr);
};
==========
"use strict";

var uniqueString = function uniqueString() {
  var arr = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : [];
  return uniqueStringWith(_fp.default.countBy(_fp.default.identity), arr);
};let arr = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : [];
==========
"use strict";

var arr = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : [];var _fp = _interopRequireDefault(require("lodash/fp"));
==========
"use strict";

var _fp = _interopRequireDefault(require("lodash/fp"));var _conversion = require("./conversion");
==========
"use strict";

var _conversion = require("./conversion");var _lang = require("./lang");
==========
"use strict";

var _lang = require("./lang");// Core
let aspect = _ref => {
  let {
    name = 'aspect',
    init = _fp.default.noop,
    after = _fp.default.noop,
    before = _fp.default.noop,
    always = _fp.default.noop,
    onError = _lang.throws // ?: interceptParams, interceptResult, wrap

  } = _ref;
  return f => {
    let {
      state = {}
    } = f;
    init(state); // Trick to set function.name of anonymous function

    let x = {
      [name]() {
        for (var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++) {
          args[_key] = arguments[_key];
        }

        let result;
        let error;
        return Promise.resolve().then(() => before(args, state)).then(() => f(...args)).then(r => {
          result = r;
        }).then(() => after(result, state, args)).catch(e => onError(e, state, args)).catch(e => {
          error = e;
        }).then(() => always(state, args)).then(() => {
          if (error) throw error;
        }).then(() => result);
      }

    };
    x[name].state = state;
    return x[name];
  };
};
==========
"use strict";

// Core
var aspect = function aspect(_ref) {
  var _ref$name = _ref.name,
      name = _ref$name === void 0 ? 'aspect' : _ref$name,
      _ref$init = _ref.init,
      init = _ref$init === void 0 ? _fp.default.noop : _ref$init,
      _ref$after = _ref.after,
      after = _ref$after === void 0 ? _fp.default.noop : _ref$after,
      _ref$before = _ref.before,
      before = _ref$before === void 0 ? _fp.default.noop : _ref$before,
      _ref$always = _ref.always,
      always = _ref$always === void 0 ? _fp.default.noop : _ref$always,
      _ref$onError = _ref.onError,
      onError = _ref$onError === void 0 ? _lang.throws : _ref$onError;
  return function (f) {
    var _f$state = f.state,
        state = _f$state === void 0 ? {} : _f$state;
    init(state); // Trick to set function.name of anonymous function

    var x = {
      [name]() {
        for (var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++) {
          args[_key] = arguments[_key];
        }

        var result;
        var error;
        return Promise.resolve().then(function () {
          return before(args, state);
        }).then(function () {
          return f.apply(void 0, args);
        }).then(function (r) {
          result = r;
        }).then(function () {
          return after(result, state, args);
        }).catch(function (e) {
          return onError(e, state, args);
        }).catch(function (e) {
          error = e;
        }).then(function () {
          return always(state, args);
        }).then(function () {
          if (error) throw error;
        }).then(function () {
          return result;
        });
      }

    };
    x[name].state = state;
    return x[name];
  };
};let {
  name = 'aspect',
  init = _fp.default.noop,
  after = _fp.default.noop,
  before = _fp.default.noop,
  always = _fp.default.noop,
  onError = _lang.throws // ?: interceptParams, interceptResult, wrap

} = _ref;
==========
"use strict";

var _ref2 = _ref,
    _ref2$name = _ref2.name,
    name = _ref2$name === void 0 ? 'aspect' : _ref2$name,
    _ref2$init = _ref2.init,
    init = _ref2$init === void 0 ? _fp.default.noop : _ref2$init,
    _ref2$after = _ref2.after,
    after = _ref2$after === void 0 ? _fp.default.noop : _ref2$after,
    _ref2$before = _ref2.before,
    before = _ref2$before === void 0 ? _fp.default.noop : _ref2$before,
    _ref2$always = _ref2.always,
    always = _ref2$always === void 0 ? _fp.default.noop : _ref2$always,
    _ref2$onError = _ref2.onError,
    onError = _ref2$onError === void 0 ? _lang.throws : _ref2$onError;let {
  state = {}
} = f;
==========
"use strict";

var _f = f,
    _f$state = _f.state,
    state = _f$state === void 0 ? {} : _f$state;// Trick to set function.name of anonymous function
let x = {
  [name]() {
    for (var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++) {
      args[_key] = arguments[_key];
    }

    let result;
    let error;
    return Promise.resolve().then(() => before(args, state)).then(() => f(...args)).then(r => {
      result = r;
    }).then(() => after(result, state, args)).catch(e => onError(e, state, args)).catch(e => {
      error = e;
    }).then(() => always(state, args)).then(() => {
      if (error) throw error;
    }).then(() => result);
  }

};
==========
"use strict";

// Trick to set function.name of anonymous function
var x = {
  [name]() {
    for (var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++) {
      args[_key] = arguments[_key];
    }

    var result;
    var error;
    return Promise.resolve().then(function () {
      return before(args, state);
    }).then(function () {
      return f.apply(void 0, args);
    }).then(function (r) {
      result = r;
    }).then(function () {
      return after(result, state, args);
    }).catch(function (e) {
      return onError(e, state, args);
    }).catch(function (e) {
      error = e;
    }).then(function () {
      return always(state, args);
    }).then(function () {
      if (error) throw error;
    }).then(function () {
      return result;
    });
  }

};var _len = arguments.length,
    args = new Array(_len),
    _key = 0;
==========
"use strict";

var _len = arguments.length,
    args = new Array(_len),
    _key = 0;let result;
==========
"use strict";

var result;let error;
==========
"use strict";

var error;let aspectSync = _ref2 => {
  let {
    name = 'aspect',
    init = _fp.default.noop,
    after = _fp.default.noop,
    before = _fp.default.noop,
    always = _fp.default.noop,
    onError = _lang.throws // ?: interceptParams, interceptResult, wrap

  } = _ref2;
  return f => {
    let {
      state = {}
    } = f;
    init(state); // Trick to set function.name of anonymous function

    let x = {
      [name]() {
        for (var _len2 = arguments.length, args = new Array(_len2), _key2 = 0; _key2 < _len2; _key2++) {
          args[_key2] = arguments[_key2];
        }

        try {
          before(args, state);
          let result = f(...args);
          after(result, state, args);
          return result;
        } catch (e) {
          onError(e, state, args);
          throw e;
        } finally {
          always(state, args);
        }
      }

    };
    x[name].state = state;
    return x[name];
  };
}; // Example Aspects
==========
"use strict";

var aspectSync = function aspectSync(_ref2) {
  var _ref2$name = _ref2.name,
      name = _ref2$name === void 0 ? 'aspect' : _ref2$name,
      _ref2$init = _ref2.init,
      init = _ref2$init === void 0 ? _fp.default.noop : _ref2$init,
      _ref2$after = _ref2.after,
      after = _ref2$after === void 0 ? _fp.default.noop : _ref2$after,
      _ref2$before = _ref2.before,
      before = _ref2$before === void 0 ? _fp.default.noop : _ref2$before,
      _ref2$always = _ref2.always,
      always = _ref2$always === void 0 ? _fp.default.noop : _ref2$always,
      _ref2$onError = _ref2.onError,
      onError = _ref2$onError === void 0 ? _lang.throws : _ref2$onError;
  return function (f) {
    var _f$state = f.state,
        state = _f$state === void 0 ? {} : _f$state;
    init(state); // Trick to set function.name of anonymous function

    var x = {
      [name]() {
        for (var _len2 = arguments.length, args = new Array(_len2), _key2 = 0; _key2 < _len2; _key2++) {
          args[_key2] = arguments[_key2];
        }

        try {
          before(args, state);
          var result = f.apply(void 0, args);
          after(result, state, args);
          return result;
        } catch (e) {
          onError(e, state, args);
          throw e;
        } finally {
          always(state, args);
        }
      }

    };
    x[name].state = state;
    return x[name];
  };
}; // Example Aspectslet {
  name = 'aspect',
  init = _fp.default.noop,
  after = _fp.default.noop,
  before = _fp.default.noop,
  always = _fp.default.noop,
  onError = _lang.throws // ?: interceptParams, interceptResult, wrap

} = _ref2;
==========
"use strict";

var _ref = _ref2,
    _ref$name = _ref.name,
    name = _ref$name === void 0 ? 'aspect' : _ref$name,
    _ref$init = _ref.init,
    init = _ref$init === void 0 ? _fp.default.noop : _ref$init,
    _ref$after = _ref.after,
    after = _ref$after === void 0 ? _fp.default.noop : _ref$after,
    _ref$before = _ref.before,
    before = _ref$before === void 0 ? _fp.default.noop : _ref$before,
    _ref$always = _ref.always,
    always = _ref$always === void 0 ? _fp.default.noop : _ref$always,
    _ref$onError = _ref.onError,
    onError = _ref$onError === void 0 ? _lang.throws : _ref$onError;let {
  state = {}
} = f;
==========
"use strict";

var _f = f,
    _f$state = _f.state,
    state = _f$state === void 0 ? {} : _f$state;// Trick to set function.name of anonymous function
let x = {
  [name]() {
    for (var _len2 = arguments.length, args = new Array(_len2), _key2 = 0; _key2 < _len2; _key2++) {
      args[_key2] = arguments[_key2];
    }

    try {
      before(args, state);
      let result = f(...args);
      after(result, state, args);
      return result;
    } catch (e) {
      onError(e, state, args);
      throw e;
    } finally {
      always(state, args);
    }
  }

};
==========
"use strict";

// Trick to set function.name of anonymous function
var x = {
  [name]() {
    for (var _len2 = arguments.length, args = new Array(_len2), _key2 = 0; _key2 < _len2; _key2++) {
      args[_key2] = arguments[_key2];
    }

    try {
      before(args, state);
      var result = f.apply(void 0, args);
      after(result, state, args);
      return result;
    } catch (e) {
      onError(e, state, args);
      throw e;
    } finally {
      always(state, args);
    }
  }

};var _len2 = arguments.length,
    args = new Array(_len2),
    _key2 = 0;
==========
"use strict";

var _len2 = arguments.length,
    args = new Array(_len2),
    _key2 = 0;let result = f(...args);
==========
"use strict";

function _toConsumableArray(arr) { return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread(); }

function _nonIterableSpread() { throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }

function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }

function _iterableToArray(iter) { if (typeof Symbol !== "undefined" && iter[Symbol.iterator] != null || iter["@@iterator"] != null) return Array.from(iter); }

function _arrayWithoutHoles(arr) { if (Array.isArray(arr)) return _arrayLikeToArray(arr); }

function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }

var result = f.apply(void 0, _toConsumableArray(args));// Example Aspects
let logs = function () {
  let extend = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : _conversion.defaultsOn;
  return aspect({
    init: extend({
      logs: []
    }),
    after: (result, state) => state.logs.push(result),
    name: 'logs'
  });
};
==========
"use strict";

// Example Aspects
var logs = function logs() {
  var extend = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : _conversion.defaultsOn;
  return aspect({
    init: extend({
      logs: []
    }),
    after: function after(result, state) {
      return state.logs.push(result);
    },
    name: 'logs'
  });
};let extend = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : _conversion.defaultsOn;
==========
"use strict";

var extend = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : _conversion.defaultsOn;let error = function () {
  let extend = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : _conversion.defaultsOn;
  return aspect({
    init: extend({
      error: null
    }),
    onError: (0, _conversion.setOn)('error'),
    name: 'error'
  });
};
==========
"use strict";

var error = function error() {
  var extend = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : _conversion.defaultsOn;
  return aspect({
    init: extend({
      error: null
    }),
    onError: (0, _conversion.setOn)('error'),
    name: 'error'
  });
};let extend = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : _conversion.defaultsOn;
==========
"use strict";

var extend = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : _conversion.defaultsOn;let errors = function () {
  let extend = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : _conversion.defaultsOn;
  return aspect({
    init: extend({
      errors: []
    }),
    onError: (e, state) => state.errors.push(e),
    name: 'errors'
  });
};
==========
"use strict";

var errors = function errors() {
  var extend = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : _conversion.defaultsOn;
  return aspect({
    init: extend({
      errors: []
    }),
    onError: function onError(e, state) {
      return state.errors.push(e);
    },
    name: 'errors'
  });
};let extend = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : _conversion.defaultsOn;
==========
"use strict";

var extend = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : _conversion.defaultsOn;let status = function () {
  let extend = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : _conversion.defaultsOn;
  return aspect({
    init: extend({
      status: null,
      processing: false,
      succeeded: false,
      failed: false,

      // Computed get/set properties don't work, probably because lodash extend methods don't support copying them
      setStatus(x) {
        this.status = x;
        this.failed = x === 'failed';
        this.succeeded = x === 'succeeded';
        this.processing = x === 'processing';
      }

    }),

    before(params, state) {
      state.setStatus('processing');
    },

    after(result, state) {
      state.setStatus('succeeded');
    },

    onError: (0, _lang.tapError)((e, state) => {
      state.setStatus('failed');
    }),
    name: 'status'
  });
};
==========
"use strict";

var status = function status() {
  var extend = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : _conversion.defaultsOn;
  return aspect({
    init: extend({
      status: null,
      processing: false,
      succeeded: false,
      failed: false,

      // Computed get/set properties don't work, probably because lodash extend methods don't support copying them
      setStatus(x) {
        this.status = x;
        this.failed = x === 'failed';
        this.succeeded = x === 'succeeded';
        this.processing = x === 'processing';
      }

    }),

    before(params, state) {
      state.setStatus('processing');
    },

    after(result, state) {
      state.setStatus('succeeded');
    },

    onError: (0, _lang.tapError)(function (e, state) {
      state.setStatus('failed');
    }),
    name: 'status'
  });
};let extend = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : _conversion.defaultsOn;
==========
"use strict";

var extend = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : _conversion.defaultsOn;let clearStatus = function () {
  let timeout = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : 500;
  return aspect({
    always(state) {
      if (timeout !== null) {
        setTimeout(() => {
          state.setStatus(null);
        }, timeout);
      }
    },

    name: 'clearStatus'
  });
}; // This is a function just for consistency
==========
"use strict";

var clearStatus = function clearStatus() {
  var timeout = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : 500;
  return aspect({
    always(state) {
      if (timeout !== null) {
        setTimeout(function () {
          state.setStatus(null);
        }, timeout);
      }
    },

    name: 'clearStatus'
  });
}; // This is a function just for consistencylet timeout = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : 500;
==========
"use strict";

var timeout = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : 500;// This is a function just for consistency
let concurrency = () => aspect({
  before(params, state) {
    if (state.processing) {
      throw Error('Concurrent Runs Not Allowed');
    }
  },

  name: 'concurrency'
});
==========
"use strict";

// This is a function just for consistency
var concurrency = function concurrency() {
  return aspect({
    before(params, state) {
      if (state.processing) {
        throw Error('Concurrent Runs Not Allowed');
      }
    },

    name: 'concurrency'
  });
};let command = (extend, timeout) => _fp.default.flow(status(extend), clearStatus(timeout), concurrency(extend), error(extend));
==========
"use strict";

var command = function command(extend, timeout) {
  return _fp.default.flow(status(extend), clearStatus(timeout), concurrency(extend), error(extend));
};let deprecate = (subject, version, alternative) => aspectSync({
  before: () => console.warn(`\`${subject}\` is deprecated${version ? ` as of ${version}` : ''}${alternative ? ` in favor of \`${alternative}\`` : ''} ${_fp.default.trim((Error().stack || '').split('\n')[3])}`)
});
==========
"use strict";

var deprecate = function deprecate(subject, version, alternative) {
  return aspectSync({
    before: function before() {
      return console.warn("`".concat(subject, "` is deprecated").concat(version ? " as of ".concat(version) : '').concat(alternative ? " in favor of `".concat(alternative, "`") : '', " ").concat(_fp.default.trim((Error().stack || '').split('\n')[3])));
    }
  });
};let aspects = {
  logs,
  error,
  errors,
  status,
  deprecate,
  clearStatus,
  concurrency,
  command
};
==========
"use strict";

var aspects = {
  logs,
  error,
  errors,
  status,
  deprecate,
  clearStatus,
  concurrency,
  command
};var _fp = _interopRequireDefault(require("lodash/fp"));
==========
"use strict";

var _fp = _interopRequireDefault(require("lodash/fp"));var _aspect = require("./aspect");
==========
"use strict";

var _aspect = require("./aspect");const noRearg = _fp.default.convert({
  rearg: false
});
==========
"use strict";

var noRearg = _fp.default.convert({
  rearg: false
});const mutable = _fp.default.convert({
  immutable: false
});
==========
"use strict";

var mutable = _fp.default.convert({
  immutable: false
});const noCap = _fp.default.convert({
  cap: false
}); // Flips
// ----------
==========
"use strict";

var noCap = _fp.default.convert({
  cap: false
}); // Flips
// ----------// Flips
// ----------
const getIn = noRearg.get;
==========
"use strict";

// Flips
// ----------
var getIn = noRearg.get;const hasIn = noRearg.has;
==========
"use strict";

var hasIn = noRearg.has;const pickIn = noRearg.pick;
==========
"use strict";

var pickIn = noRearg.pick;const includesIn = noRearg.includes;
==========
"use strict";

var includesIn = noRearg.includes;const inversions = _fp.default.mapKeys(k => `${k}In`, noRearg); // Mutables
// ----------
==========
"use strict";

var inversions = _fp.default.mapKeys(function (k) {
  return "".concat(k, "In");
}, noRearg); // Mutables
// ----------// Mutables
// ----------
const extendOn = mutable.extend;
==========
"use strict";

// Mutables
// ----------
var extendOn = mutable.extend;const defaultsOn = mutable.defaults;
==========
"use strict";

var defaultsOn = mutable.defaults;const mergeOn = mutable.merge;
==========
"use strict";

var mergeOn = mutable.merge;const setOn = mutable.set; // Curry required until https://github.com/lodash/lodash/issues/3440 is resolved
==========
"use strict";

var setOn = mutable.set; // Curry required until https://github.com/lodash/lodash/issues/3440 is resolved// Curry required until https://github.com/lodash/lodash/issues/3440 is resolved
const unsetOn = _fp.default.curryN(2, mutable.unset);
==========
"use strict";

// Curry required until https://github.com/lodash/lodash/issues/3440 is resolved
var unsetOn = _fp.default.curryN(2, mutable.unset);const pullOn = mutable.pull;
==========
"use strict";

var pullOn = mutable.pull;const updateOn = mutable.update; // Uncaps
// ------
// Un-prefixed Deprecated
==========
"use strict";

var updateOn = mutable.update; // Uncaps
// ------
// Un-prefixed Deprecated// Uncaps
// ------
// Un-prefixed Deprecated
const reduce = _aspect.aspects.deprecate('reduce', '1.28.0', 'reduceIndexed')(noCap.reduce);
==========
"use strict";

// Uncaps
// ------
// Un-prefixed Deprecated
var reduce = _aspect.aspects.deprecate('reduce', '1.28.0', 'reduceIndexed')(noCap.reduce);const mapValues = _aspect.aspects.deprecate('mapValues', '1.28.0', 'mapValuesIndexed')(noCap.mapValues);
==========
"use strict";

var mapValues = _aspect.aspects.deprecate('mapValues', '1.28.0', 'mapValuesIndexed')(noCap.mapValues);const each = _aspect.aspects.deprecate('each', '1.28.0', 'eachIndexed')(noCap.each);
==========
"use strict";

var each = _aspect.aspects.deprecate('each', '1.28.0', 'eachIndexed')(noCap.each);const mapIndexed = noCap.map;
==========
"use strict";

var mapIndexed = noCap.map;const findIndexed = noCap.find;
==========
"use strict";

var findIndexed = noCap.find;const eachIndexed = noCap.each;
==========
"use strict";

var eachIndexed = noCap.each;const reduceIndexed = noCap.reduce;
==========
"use strict";

var reduceIndexed = noCap.reduce;const pickByIndexed = noCap.pickBy;
==========
"use strict";

var pickByIndexed = noCap.pickBy;const mapValuesIndexed = noCap.mapValues;
==========
"use strict";

var mapValuesIndexed = noCap.mapValues;var _fp = _interopRequireDefault(require("lodash/fp"));
==========
"use strict";

var _fp = _interopRequireDefault(require("lodash/fp"));var _tree = require("./tree");
==========
"use strict";

var _tree = require("./tree");let throws = x => {
  throw x;
};
==========
"use strict";

var throws = function throws(x) {
  throw x;
};let tapError = f => function (e) {
  for (var _len = arguments.length, args = new Array(_len > 1 ? _len - 1 : 0), _key = 1; _key < _len; _key++) {
    args[_key - 1] = arguments[_key];
  }

  f(e, ...args);
  throw e;
};
==========
"use strict";

var tapError = function tapError(f) {
  return function (e) {
    for (var _len = arguments.length, args = new Array(_len > 1 ? _len - 1 : 0), _key = 1; _key < _len; _key++) {
      args[_key - 1] = arguments[_key];
    }

    f.apply(void 0, [e].concat(args));
    throw e;
  };
};var _len = arguments.length,
    args = new Array(_len > 1 ? _len - 1 : 0),
    _key = 1;
==========
"use strict";

var _len = arguments.length,
    args = new Array(_len > 1 ? _len - 1 : 0),
    _key = 1;let isNotNil = _fp.default.negate(_fp.default.isNil);
==========
"use strict";

var isNotNil = _fp.default.negate(_fp.default.isNil);let exists = isNotNil;
==========
"use strict";

var exists = isNotNil;let isMultiple = x => (x || []).length > 1;
==========
"use strict";

var isMultiple = function isMultiple(x) {
  return (x || []).length > 1;
};let append = _fp.default.curry((x, y) => y + x); // True for everything except null, undefined, '', [], and {}
==========
"use strict";

var append = _fp.default.curry(function (x, y) {
  return y + x;
}); // True for everything except null, undefined, '', [], and {}// True for everything except null, undefined, '', [], and {}
let isBlank = _fp.default.overSome([_fp.default.isNil, _fp.default.isEqual(''), _fp.default.isEqual([]), _fp.default.isEqual({})]);
==========
"use strict";

// True for everything except null, undefined, '', [], and {}
var isBlank = _fp.default.overSome([_fp.default.isNil, _fp.default.isEqual(''), _fp.default.isEqual([]), _fp.default.isEqual({})]);let isNotBlank = _fp.default.negate(isBlank);
==========
"use strict";

var isNotBlank = _fp.default.negate(isBlank);let isBlankDeep = combinator => x => combinator(isBlank, (0, _tree.tree)().leaves(x));
==========
"use strict";

var isBlankDeep = function isBlankDeep(combinator) {
  return function (x) {
    return combinator(isBlank, (0, _tree.tree)().leaves(x));
  };
};var _fp = _interopRequireDefault(require("lodash/fp"));
==========
"use strict";

var _fp = _interopRequireDefault(require("lodash/fp"));var _function = require("./function");
==========
"use strict";

var _function = require("./function");var _lang = require("./lang");
==========
"use strict";

var _lang = require("./lang");// ([f, g]) -> !f(x) && !g(x)
const overNone = _fp.default.flow(_fp.default.overSome, _fp.default.negate);
==========
"use strict";

// ([f, g]) -> !f(x) && !g(x)
var overNone = _fp.default.flow(_fp.default.overSome, _fp.default.negate);let boolIteratee = x => _fp.default.isBoolean(x) || _fp.default.isNil(x) ? () => x : _fp.default.iteratee(x); // Port from Ramda
==========
"use strict";

var boolIteratee = function boolIteratee(x) {
  return _fp.default.isBoolean(x) || _fp.default.isNil(x) ? function () {
    return x;
  } : _fp.default.iteratee(x);
}; // Port from Ramda// Port from Ramda
let ifElse = _fp.default.curry((condition, onTrue, onFalse, x) => boolIteratee(condition)(x) ? (0, _function.callOrReturn)(onTrue, x) : (0, _function.callOrReturn)(onFalse, x));
==========
"use strict";

// Port from Ramda
var ifElse = _fp.default.curry(function (condition, onTrue, onFalse, x) {
  return boolIteratee(condition)(x) ? (0, _function.callOrReturn)(onTrue, x) : (0, _function.callOrReturn)(onFalse, x);
});let when = _fp.default.curry((condition, t, x) => ifElse(condition, t, _fp.default.identity, x));
==========
"use strict";

var when = _fp.default.curry(function (condition, t, x) {
  return ifElse(condition, t, _fp.default.identity, x);
});let unless = _fp.default.curry((condition, f, x) => ifElse(condition, _fp.default.identity, f, x));
==========
"use strict";

var unless = _fp.default.curry(function (condition, f, x) {
  return ifElse(condition, _fp.default.identity, f, x);
});let whenExists = when(_lang.exists);
==========
"use strict";

var whenExists = when(_lang.exists);let whenTruthy = when(Boolean);
==========
"use strict";

var whenTruthy = when(Boolean);var _fp = _interopRequireDefault(require("lodash/fp"));
==========
"use strict";

var _fp = _interopRequireDefault(require("lodash/fp"));var _array = require("./array");
==========
"use strict";

var _array = require("./array");var _collection = require("./collection");
==========
"use strict";

var _collection = require("./collection");const testRegex = _fp.default.curry((regex, str) => new RegExp(regex).test(str));
==========
"use strict";

var testRegex = _fp.default.curry(function (regex, str) {
  return new RegExp(regex).test(str);
});const makeRegex = options => text => RegExp(text, options);
==========
"use strict";

var makeRegex = function makeRegex(options) {
  return function (text) {
    return RegExp(text, options);
  };
};const makeAndTest = options => _fp.default.flow(makeRegex(options), testRegex);
==========
"use strict";

var makeAndTest = function makeAndTest(options) {
  return _fp.default.flow(makeRegex(options), testRegex);
};const anyWordToRegexp = _fp.default.flow(_fp.default.words, _fp.default.join('|'));
==========
"use strict";

var anyWordToRegexp = _fp.default.flow(_fp.default.words, _fp.default.join('|'));const wordsToRegexp = _fp.default.flow(_fp.default.words, _fp.default.map(x => `(?=.*${x}.*)`), _fp.default.join(''), x => `.*${x}.*`);
==========
"use strict";

var wordsToRegexp = _fp.default.flow(_fp.default.words, _fp.default.map(function (x) {
  return "(?=.*".concat(x, ".*)");
}), _fp.default.join(''), function (x) {
  return ".*".concat(x, ".*");
});const matchWords = _fp.default.curry((buildRegex, x) => {
  // Not inlining so that we don't create the regexp every time
  const regexp = RegExp(buildRegex(x), 'gi');
  return y => !!(y && y.match(regexp));
});
==========
"use strict";

var matchWords = _fp.default.curry(function (buildRegex, x) {
  // Not inlining so that we don't create the regexp every time
  var regexp = RegExp(buildRegex(x), 'gi');
  return function (y) {
    return !!(y && y.match(regexp));
  };
});// Not inlining so that we don't create the regexp every time
const regexp = RegExp(buildRegex(x), 'gi');
==========
"use strict";

// Not inlining so that we don't create the regexp every time
var regexp = RegExp(buildRegex(x), 'gi');const matchAllWords = matchWords(wordsToRegexp);
==========
"use strict";

var matchAllWords = matchWords(wordsToRegexp);const matchAnyWord = matchWords(anyWordToRegexp);
==========
"use strict";

var matchAnyWord = matchWords(anyWordToRegexp);const allMatches = _fp.default.curry((regexStr, str) => {
  let matched;
  const regex = new RegExp(regexStr, 'g');
  const result = [];

  while ((matched = regex.exec(str)) !== null) {
    result.push({
      text: matched[0],
      start: matched.index,
      end: regex.lastIndex
    });
  }

  return result;
});
==========
"use strict";

var allMatches = _fp.default.curry(function (regexStr, str) {
  var matched;
  var regex = new RegExp(regexStr, 'g');
  var result = [];

  while ((matched = regex.exec(str)) !== null) {
    result.push({
      text: matched[0],
      start: matched.index,
      end: regex.lastIndex
    });
  }

  return result;
});let matched;
==========
"use strict";

var matched;const regex = new RegExp(regexStr, 'g');
==========
"use strict";

var regex = new RegExp(regexStr, 'g');const result = [];
==========
"use strict";

var result = [];const postings = _fp.default.curry((regex, str) => {
  var match = regex.exec(str);
  let result = [];

  if (regex.flags.indexOf('g') < 0 && match) {
    result.push([match.index, match.index + match[0].length]);
  } else {
    while (match) {
      result.push([match.index, regex.lastIndex]);
      match = regex.exec(str);
    }
  }

  return result;
});
==========
"use strict";

var postings = _fp.default.curry(function (regex, str) {
  var match = regex.exec(str);
  var result = [];

  if (regex.flags.indexOf('g') < 0 && match) {
    result.push([match.index, match.index + match[0].length]);
  } else {
    while (match) {
      result.push([match.index, regex.lastIndex]);
      match = regex.exec(str);
    }
  }

  return result;
});var match = regex.exec(str);
==========
"use strict";

var match = regex.exec(str);let result = [];
==========
"use strict";

var result = [];const postingsForWords = _fp.default.curry((string, str) => _fp.default.reduce((result, word) => (0, _array.push)(postings(RegExp(word, 'gi'), str), result), [])(_fp.default.words(string)));
==========
"use strict";

var postingsForWords = _fp.default.curry(function (string, str) {
  return _fp.default.reduce(function (result, word) {
    return (0, _array.push)(postings(RegExp(word, 'gi'), str), result);
  }, [])(_fp.default.words(string));
});const highlightFromPostings = _fp.default.curry((start, end, postings, str) => {
  let offset = 0;

  _fp.default.each(posting => {
    str = (0, _collection.insertAtIndex)(posting[0] + offset, start, str);
    offset += start.length;
    str = (0, _collection.insertAtIndex)(posting[1] + offset, end, str);
    offset += end.length;
  }, (0, _array.mergeRanges)(postings));

  return str;
});
==========
"use strict";

var highlightFromPostings = _fp.default.curry(function (start, end, postings, str) {
  var offset = 0;

  _fp.default.each(function (posting) {
    str = (0, _collection.insertAtIndex)(posting[0] + offset, start, str);
    offset += start.length;
    str = (0, _collection.insertAtIndex)(posting[1] + offset, end, str);
    offset += end.length;
  }, (0, _array.mergeRanges)(postings));

  return str;
});let offset = 0;
==========
"use strict";

var offset = 0;const highlight = _fp.default.curry((start, end, pattern, input) => highlightFromPostings(start, end, _fp.default.isRegExp(pattern) ? postings(pattern, input) : _fp.default.flatten(postingsForWords(pattern, input)), input));
==========
"use strict";

var highlight = _fp.default.curry(function (start, end, pattern, input) {
  return highlightFromPostings(start, end, _fp.default.isRegExp(pattern) ? postings(pattern, input) : _fp.default.flatten(postingsForWords(pattern, input)), input);
});var _fp = _interopRequireDefault(require("lodash/fp"));
==========
"use strict";

var _fp = _interopRequireDefault(require("lodash/fp"));var _conversion = require("./conversion");
==========
"use strict";

var _conversion = require("./conversion");var _array = require("./array");
==========
"use strict";

var _array = require("./array");let isTraversable = x => _fp.default.isArray(x) || _fp.default.isPlainObject(x);
==========
"use strict";

var isTraversable = function isTraversable(x) {
  return _fp.default.isArray(x) || _fp.default.isPlainObject(x);
};let traverse = x => isTraversable(x) && !_fp.default.isEmpty(x) && x;
==========
"use strict";

var traverse = function traverse(x) {
  return isTraversable(x) && !_fp.default.isEmpty(x) && x;
};let walk = function () {
  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  return function (pre) {
    let post = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _fp.default.noop;
    let parents = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : [];
    let parentIndexes = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : [];
    return (tree, index) => pre(tree, index, parents, parentIndexes) || (0, _conversion.findIndexed)(walk(next)(pre, post, [tree, ...parents], [index, ...parentIndexes]), next(tree, index, parents, parentIndexes) || []) || post(tree, index, parents, parentIndexes);
  };
}; // async/await is so much cleaner but causes regeneratorRuntime shenanigans
// export let findIndexedAsync = async (f, data) => {
//   for (let key in data) {
//     if (await f(data[key], key, data)) return data[key]
//   }
// }
// The general idea here is to keep popping off key/value pairs until we hit something that matches
==========
"use strict";

function _toConsumableArray(arr) { return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread(); }

function _nonIterableSpread() { throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }

function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }

function _iterableToArray(iter) { if (typeof Symbol !== "undefined" && iter[Symbol.iterator] != null || iter["@@iterator"] != null) return Array.from(iter); }

function _arrayWithoutHoles(arr) { if (Array.isArray(arr)) return _arrayLikeToArray(arr); }

function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }

var walk = function walk() {
  var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  return function (pre) {
    var post = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _fp.default.noop;
    var parents = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : [];
    var parentIndexes = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : [];
    return function (tree, index) {
      return pre(tree, index, parents, parentIndexes) || (0, _conversion.findIndexed)(walk(next)(pre, post, [tree].concat(_toConsumableArray(parents)), [index].concat(_toConsumableArray(parentIndexes))), next(tree, index, parents, parentIndexes) || []) || post(tree, index, parents, parentIndexes);
    };
  };
}; // async/await is so much cleaner but causes regeneratorRuntime shenanigans
// export let findIndexedAsync = async (f, data) => {
//   for (let key in data) {
//     if (await f(data[key], key, data)) return data[key]
//   }
// }
// The general idea here is to keep popping off key/value pairs until we hit something that matcheslet next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
==========
"use strict";

var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;let post = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _fp.default.noop;
==========
"use strict";

var post = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _fp.default.noop;let parents = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : [];
==========
"use strict";

var parents = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : [];let parentIndexes = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : [];
==========
"use strict";

var parentIndexes = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : [];// async/await is so much cleaner but causes regeneratorRuntime shenanigans
// export let findIndexedAsync = async (f, data) => {
//   for (let key in data) {
//     if (await f(data[key], key, data)) return data[key]
//   }
// }
// The general idea here is to keep popping off key/value pairs until we hit something that matches
let findIndexedAsync = function (f, data) {
  let remaining = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : _fp.default.toPairs(data);
  if (!remaining.length) return;
  let [[key, val], ...rest] = remaining;
  return Promise.resolve(f(val, key, data)).then(result => result ? val : rest.length ? findIndexedAsync(f, data, rest) : undefined);
};
==========
"use strict";

function _slicedToArray(arr, i) { return _arrayWithHoles(arr) || _iterableToArrayLimit(arr, i) || _unsupportedIterableToArray(arr, i) || _nonIterableRest(); }

function _iterableToArrayLimit(arr, i) { var _i = arr == null ? null : typeof Symbol !== "undefined" && arr[Symbol.iterator] || arr["@@iterator"]; if (_i == null) return; var _arr = []; var _n = true; var _d = false; var _s, _e; try { for (_i = _i.call(arr); !(_n = (_s = _i.next()).done); _n = true) { _arr.push(_s.value); if (i && _arr.length === i) break; } } catch (err) { _d = true; _e = err; } finally { try { if (!_n && _i["return"] != null) _i["return"](); } finally { if (_d) throw _e; } } return _arr; }

function _toArray(arr) { return _arrayWithHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableRest(); }

function _nonIterableRest() { throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }

function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }

function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }

function _iterableToArray(iter) { if (typeof Symbol !== "undefined" && iter[Symbol.iterator] != null || iter["@@iterator"] != null) return Array.from(iter); }

function _arrayWithHoles(arr) { if (Array.isArray(arr)) return arr; }

// async/await is so much cleaner but causes regeneratorRuntime shenanigans
// export let findIndexedAsync = async (f, data) => {
//   for (let key in data) {
//     if (await f(data[key], key, data)) return data[key]
//   }
// }
// The general idea here is to keep popping off key/value pairs until we hit something that matches
var findIndexedAsync = function findIndexedAsync(f, data) {
  var remaining = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : _fp.default.toPairs(data);
  if (!remaining.length) return;

  var _remaining = _toArray(remaining),
      _remaining$ = _slicedToArray(_remaining[0], 2),
      key = _remaining$[0],
      val = _remaining$[1],
      rest = _remaining.slice(1);

  return Promise.resolve(f(val, key, data)).then(function (result) {
    return result ? val : rest.length ? findIndexedAsync(f, data, rest) : undefined;
  });
};let remaining = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : _fp.default.toPairs(data);
==========
"use strict";

var remaining = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : _fp.default.toPairs(data);let [[key, val], ...rest] = remaining;
==========
"use strict";

function _slicedToArray(arr, i) { return _arrayWithHoles(arr) || _iterableToArrayLimit(arr, i) || _unsupportedIterableToArray(arr, i) || _nonIterableRest(); }

function _iterableToArrayLimit(arr, i) { var _i = arr == null ? null : typeof Symbol !== "undefined" && arr[Symbol.iterator] || arr["@@iterator"]; if (_i == null) return; var _arr = []; var _n = true; var _d = false; var _s, _e; try { for (_i = _i.call(arr); !(_n = (_s = _i.next()).done); _n = true) { _arr.push(_s.value); if (i && _arr.length === i) break; } } catch (err) { _d = true; _e = err; } finally { try { if (!_n && _i["return"] != null) _i["return"](); } finally { if (_d) throw _e; } } return _arr; }

function _toArray(arr) { return _arrayWithHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableRest(); }

function _nonIterableRest() { throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }

function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }

function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }

function _iterableToArray(iter) { if (typeof Symbol !== "undefined" && iter[Symbol.iterator] != null || iter["@@iterator"] != null) return Array.from(iter); }

function _arrayWithHoles(arr) { if (Array.isArray(arr)) return arr; }

var _remaining = remaining,
    _remaining2 = _toArray(_remaining),
    _remaining2$ = _slicedToArray(_remaining2[0], 2),
    key = _remaining2$[0],
    val = _remaining2$[1],
    rest = _remaining2.slice(1);let walkAsync = function () {
  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  return function (pre) {
    let post = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _fp.default.noop;
    let parents = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : [];
    let parentIndexes = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : [];
    return (tree, index) => Promise.resolve(pre(tree, index, parents, parentIndexes)).then(preResult => preResult || findIndexedAsync(walkAsync(next)(pre, post, [tree, ...parents], [index, ...parentIndexes]), next(tree, index, parents, parentIndexes) || [])).then(stepResult => stepResult || post(tree, index, parents, parentIndexes));
  };
};
==========
"use strict";

function _toConsumableArray(arr) { return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread(); }

function _nonIterableSpread() { throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }

function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }

function _iterableToArray(iter) { if (typeof Symbol !== "undefined" && iter[Symbol.iterator] != null || iter["@@iterator"] != null) return Array.from(iter); }

function _arrayWithoutHoles(arr) { if (Array.isArray(arr)) return _arrayLikeToArray(arr); }

function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }

var walkAsync = function walkAsync() {
  var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  return function (pre) {
    var post = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _fp.default.noop;
    var parents = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : [];
    var parentIndexes = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : [];
    return function (tree, index) {
      return Promise.resolve(pre(tree, index, parents, parentIndexes)).then(function (preResult) {
        return preResult || findIndexedAsync(walkAsync(next)(pre, post, [tree].concat(_toConsumableArray(parents)), [index].concat(_toConsumableArray(parentIndexes))), next(tree, index, parents, parentIndexes) || []);
      }).then(function (stepResult) {
        return stepResult || post(tree, index, parents, parentIndexes);
      });
    };
  };
};let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
==========
"use strict";

var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;let post = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _fp.default.noop;
==========
"use strict";

var post = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _fp.default.noop;let parents = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : [];
==========
"use strict";

var parents = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : [];let parentIndexes = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : [];
==========
"use strict";

var parentIndexes = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : [];let transformTree = function () {
  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  return _fp.default.curry((f, x) => {
    let result = _fp.default.cloneDeep(x);

    walk(next)(f)(result);
    return result;
  });
};
==========
"use strict";

var transformTree = function transformTree() {
  var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  return _fp.default.curry(function (f, x) {
    var result = _fp.default.cloneDeep(x);

    walk(next)(f)(result);
    return result;
  });
};let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
==========
"use strict";

var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;let result = _fp.default.cloneDeep(x);
==========
"use strict";

var result = _fp.default.cloneDeep(x);let reduceTree = function () {
  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  return _fp.default.curry((f, result, tree) => {
    walk(next)(function () {
      for (var _len = arguments.length, x = new Array(_len), _key = 0; _key < _len; _key++) {
        x[_key] = arguments[_key];
      }

      result = f(result, ...x);
    })(tree);
    return result;
  });
};
==========
"use strict";

var reduceTree = function reduceTree() {
  var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  return _fp.default.curry(function (f, result, tree) {
    walk(next)(function () {
      for (var _len = arguments.length, x = new Array(_len), _key = 0; _key < _len; _key++) {
        x[_key] = arguments[_key];
      }

      result = f.apply(void 0, [result].concat(x));
    })(tree);
    return result;
  });
};let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
==========
"use strict";

var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;var _len = arguments.length,
    x = new Array(_len),
    _key = 0;
==========
"use strict";

var _len = arguments.length,
    x = new Array(_len),
    _key = 0;let writeProperty = function () {
  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  return (node, index, _ref) => {
    let [parent] = _ref;
    next(parent)[index] = node;
  };
};
==========
"use strict";

function _slicedToArray(arr, i) { return _arrayWithHoles(arr) || _iterableToArrayLimit(arr, i) || _unsupportedIterableToArray(arr, i) || _nonIterableRest(); }

function _nonIterableRest() { throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }

function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }

function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }

function _iterableToArrayLimit(arr, i) { var _i = arr == null ? null : typeof Symbol !== "undefined" && arr[Symbol.iterator] || arr["@@iterator"]; if (_i == null) return; var _arr = []; var _n = true; var _d = false; var _s, _e; try { for (_i = _i.call(arr); !(_n = (_s = _i.next()).done); _n = true) { _arr.push(_s.value); if (i && _arr.length === i) break; } } catch (err) { _d = true; _e = err; } finally { try { if (!_n && _i["return"] != null) _i["return"](); } finally { if (_d) throw _e; } } return _arr; }

function _arrayWithHoles(arr) { if (Array.isArray(arr)) return arr; }

var writeProperty = function writeProperty() {
  var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  return function (node, index, _ref) {
    var _ref2 = _slicedToArray(_ref, 1),
        parent = _ref2[0];

    next(parent)[index] = node;
  };
};let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
==========
"use strict";

var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;let [parent] = _ref;
==========
"use strict";

function _slicedToArray(arr, i) { return _arrayWithHoles(arr) || _iterableToArrayLimit(arr, i) || _unsupportedIterableToArray(arr, i) || _nonIterableRest(); }

function _nonIterableRest() { throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }

function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }

function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }

function _iterableToArrayLimit(arr, i) { var _i = arr == null ? null : typeof Symbol !== "undefined" && arr[Symbol.iterator] || arr["@@iterator"]; if (_i == null) return; var _arr = []; var _n = true; var _d = false; var _s, _e; try { for (_i = _i.call(arr); !(_n = (_s = _i.next()).done); _n = true) { _arr.push(_s.value); if (i && _arr.length === i) break; } } catch (err) { _d = true; _e = err; } finally { try { if (!_n && _i["return"] != null) _i["return"](); } finally { if (_d) throw _e; } } return _arr; }

function _arrayWithHoles(arr) { if (Array.isArray(arr)) return arr; }

var _ref2 = _ref,
    _ref3 = _slicedToArray(_ref2, 1),
    parent = _ref3[0];let mapTree = function () {
  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  let writeNode = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : writeProperty(next);
  return _fp.default.curry((mapper, tree) => transformTree(next)(function (node, i, parents) {
    for (var _len2 = arguments.length, args = new Array(_len2 > 3 ? _len2 - 3 : 0), _key2 = 3; _key2 < _len2; _key2++) {
      args[_key2 - 3] = arguments[_key2];
    }

    if (parents.length) writeNode(mapper(node, i, parents, ...args), i, parents, ...args);
  })(mapper(tree)) // run mapper on root, and skip root in traversal
  );
};
==========
"use strict";

var mapTree = function mapTree() {
  var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  var writeNode = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : writeProperty(next);
  return _fp.default.curry(function (mapper, tree) {
    return transformTree(next)(function (node, i, parents) {
      for (var _len2 = arguments.length, args = new Array(_len2 > 3 ? _len2 - 3 : 0), _key2 = 3; _key2 < _len2; _key2++) {
        args[_key2 - 3] = arguments[_key2];
      }

      if (parents.length) writeNode.apply(void 0, [mapper.apply(void 0, [node, i, parents].concat(args)), i, parents].concat(args));
    })(mapper(tree));
  } // run mapper on root, and skip root in traversal
  );
};let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
==========
"use strict";

var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;let writeNode = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : writeProperty(next);
==========
"use strict";

var writeNode = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : writeProperty(next);var _len2 = arguments.length,
    args = new Array(_len2 > 3 ? _len2 - 3 : 0),
    _key2 = 3;
==========
"use strict";

var _len2 = arguments.length,
    args = new Array(_len2 > 3 ? _len2 - 3 : 0),
    _key2 = 3;let mapTreeLeaves = function () {
  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  let writeNode = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : writeProperty(next);
  return _fp.default.curry((mapper, tree) => // this unless wrapping can be done in user land, this is pure convenience
  // mapTree(next, writeNode)(F.unless(next, mapper), tree)
  mapTree(next, writeNode)(node => next(node) ? node : mapper(node), tree));
};
==========
"use strict";

var mapTreeLeaves = function mapTreeLeaves() {
  var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  var writeNode = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : writeProperty(next);
  return _fp.default.curry(function (mapper, tree) {
    return (// this unless wrapping can be done in user land, this is pure convenience
      // mapTree(next, writeNode)(F.unless(next, mapper), tree)
      mapTree(next, writeNode)(function (node) {
        return next(node) ? node : mapper(node);
      }, tree)
    );
  });
};let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
==========
"use strict";

var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;let writeNode = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : writeProperty(next);
==========
"use strict";

var writeNode = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : writeProperty(next);let treeToArrayBy = function () {
  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  return _fp.default.curry((fn, tree) => reduceTree(next)(function (r) {
    for (var _len3 = arguments.length, args = new Array(_len3 > 1 ? _len3 - 1 : 0), _key3 = 1; _key3 < _len3; _key3++) {
      args[_key3 - 1] = arguments[_key3];
    }

    return (0, _array.push)(fn(...args), r);
  }, [], tree));
};
==========
"use strict";

var treeToArrayBy = function treeToArrayBy() {
  var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  return _fp.default.curry(function (fn, tree) {
    return reduceTree(next)(function (r) {
      for (var _len3 = arguments.length, args = new Array(_len3 > 1 ? _len3 - 1 : 0), _key3 = 1; _key3 < _len3; _key3++) {
        args[_key3 - 1] = arguments[_key3];
      }

      return (0, _array.push)(fn.apply(void 0, args), r);
    }, [], tree);
  });
};let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
==========
"use strict";

var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;var _len3 = arguments.length,
    args = new Array(_len3 > 1 ? _len3 - 1 : 0),
    _key3 = 1;
==========
"use strict";

var _len3 = arguments.length,
    args = new Array(_len3 > 1 ? _len3 - 1 : 0),
    _key3 = 1;let treeToArray = function () {
  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  return treeToArrayBy(next)(x => x);
}; // This could reuse treeToArrayBy and just reject traversable elements after, but this is more efficient
// We can potentially unify these with tree transducers
==========
"use strict";

var treeToArray = function treeToArray() {
  var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  return treeToArrayBy(next)(function (x) {
    return x;
  });
}; // This could reuse treeToArrayBy and just reject traversable elements after, but this is more efficient
// We can potentially unify these with tree transducerslet next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
==========
"use strict";

var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;// This could reuse treeToArrayBy and just reject traversable elements after, but this is more efficient
// We can potentially unify these with tree transducers
let leavesBy = function () {
  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  return _fp.default.curry((fn, tree) => reduceTree(next)(function (r, node) {
    for (var _len4 = arguments.length, args = new Array(_len4 > 2 ? _len4 - 2 : 0), _key4 = 2; _key4 < _len4; _key4++) {
      args[_key4 - 2] = arguments[_key4];
    }

    return next(node) ? r : (0, _array.push)(fn(node, ...args), r);
  }, [], tree));
};
==========
"use strict";

// This could reuse treeToArrayBy and just reject traversable elements after, but this is more efficient
// We can potentially unify these with tree transducers
var leavesBy = function leavesBy() {
  var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  return _fp.default.curry(function (fn, tree) {
    return reduceTree(next)(function (r, node) {
      for (var _len4 = arguments.length, args = new Array(_len4 > 2 ? _len4 - 2 : 0), _key4 = 2; _key4 < _len4; _key4++) {
        args[_key4 - 2] = arguments[_key4];
      }

      return next(node) ? r : (0, _array.push)(fn.apply(void 0, [node].concat(args)), r);
    }, [], tree);
  });
};let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
==========
"use strict";

var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;var _len4 = arguments.length,
    args = new Array(_len4 > 2 ? _len4 - 2 : 0),
    _key4 = 2;
==========
"use strict";

var _len4 = arguments.length,
    args = new Array(_len4 > 2 ? _len4 - 2 : 0),
    _key4 = 2;let leaves = function () {
  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  return leavesBy(next)(x => x);
};
==========
"use strict";

var leaves = function leaves() {
  var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  return leavesBy(next)(function (x) {
    return x;
  });
};let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
==========
"use strict";

var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;let treeLookup = function () {
  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  let buildIteratee = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _fp.default.identity;
  return _fp.default.curry((path, tree) => _fp.default.reduce((tree, path) => (0, _conversion.findIndexed)(buildIteratee(path), next(tree)), tree, path));
};
==========
"use strict";

var treeLookup = function treeLookup() {
  var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  var buildIteratee = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _fp.default.identity;
  return _fp.default.curry(function (path, tree) {
    return _fp.default.reduce(function (tree, path) {
      return (0, _conversion.findIndexed)(buildIteratee(path), next(tree));
    }, tree, path);
  });
};let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
==========
"use strict";

var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;let buildIteratee = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _fp.default.identity;
==========
"use strict";

var buildIteratee = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _fp.default.identity;let keyTreeByWith = function () {
  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  return _fp.default.curry((transformer, groupIteratee, x) => _fp.default.flow(treeToArrayBy(next)(_fp.default.iteratee(groupIteratee)), _fp.default.uniq, _fp.default.keyBy(_fp.default.identity), _fp.default.mapValues(group => transformTree(next)(node => {
    let matches = _fp.default.iteratee(groupIteratee)(node) === group;
    transformer(node, matches, group);
  }, x)))(x));
}; // Flat Tree
==========
"use strict";

var keyTreeByWith = function keyTreeByWith() {
  var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  return _fp.default.curry(function (transformer, groupIteratee, x) {
    return _fp.default.flow(treeToArrayBy(next)(_fp.default.iteratee(groupIteratee)), _fp.default.uniq, _fp.default.keyBy(_fp.default.identity), _fp.default.mapValues(function (group) {
      return transformTree(next)(function (node) {
        var matches = _fp.default.iteratee(groupIteratee)(node) === group;
        transformer(node, matches, group);
      }, x);
    }))(x);
  });
}; // Flat Treelet next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
==========
"use strict";

var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;let matches = _fp.default.iteratee(groupIteratee)(node) === group;
==========
"use strict";

var matches = _fp.default.iteratee(groupIteratee)(node) === group;// Flat Tree
let treeKeys = (x, i, xs, is) => [i, ...is];
==========
"use strict";

function _toConsumableArray(arr) { return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread(); }

function _nonIterableSpread() { throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }

function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }

function _iterableToArray(iter) { if (typeof Symbol !== "undefined" && iter[Symbol.iterator] != null || iter["@@iterator"] != null) return Array.from(iter); }

function _arrayWithoutHoles(arr) { if (Array.isArray(arr)) return _arrayLikeToArray(arr); }

function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }

// Flat Tree
var treeKeys = function treeKeys(x, i, xs, is) {
  return [i].concat(_toConsumableArray(is));
};let treeValues = (x, i, xs) => [x, ...xs];
==========
"use strict";

function _toConsumableArray(arr) { return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread(); }

function _nonIterableSpread() { throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }

function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }

function _iterableToArray(iter) { if (typeof Symbol !== "undefined" && iter[Symbol.iterator] != null || iter["@@iterator"] != null) return Array.from(iter); }

function _arrayWithoutHoles(arr) { if (Array.isArray(arr)) return _arrayLikeToArray(arr); }

function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }

var treeValues = function treeValues(x, i, xs) {
  return [x].concat(_toConsumableArray(xs));
};let treePath = function () {
  let build = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : treeKeys;
  let encoder = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _array.dotEncoder;
  return function () {
    return (encoder.encode || encoder)(build(...arguments).reverse());
  };
};
==========
"use strict";

var treePath = function treePath() {
  var build = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : treeKeys;
  var encoder = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _array.dotEncoder;
  return function () {
    return (encoder.encode || encoder)(build.apply(void 0, arguments).reverse());
  };
};let build = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : treeKeys;
==========
"use strict";

var build = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : treeKeys;let encoder = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _array.dotEncoder;
==========
"use strict";

var encoder = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _array.dotEncoder;let propTreePath = prop => treePath(_fp.default.flow(treeValues, _fp.default.map(prop)), _array.slashEncoder);
==========
"use strict";

var propTreePath = function propTreePath(prop) {
  return treePath(_fp.default.flow(treeValues, _fp.default.map(prop)), _array.slashEncoder);
};let flattenTree = function () {
  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  return function () {
    let buildPath = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : treePath();
    return reduceTree(next)(function (result, node) {
      for (var _len5 = arguments.length, x = new Array(_len5 > 2 ? _len5 - 2 : 0), _key5 = 2; _key5 < _len5; _key5++) {
        x[_key5 - 2] = arguments[_key5];
      }

      return _fp.default.set([buildPath(node, ...x)], node, result);
    }, {});
  };
};
==========
"use strict";

var flattenTree = function flattenTree() {
  var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  return function () {
    var buildPath = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : treePath();
    return reduceTree(next)(function (result, node) {
      for (var _len5 = arguments.length, x = new Array(_len5 > 2 ? _len5 - 2 : 0), _key5 = 2; _key5 < _len5; _key5++) {
        x[_key5 - 2] = arguments[_key5];
      }

      return _fp.default.set([buildPath.apply(void 0, [node].concat(x))], node, result);
    }, {});
  };
};let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
==========
"use strict";

var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;let buildPath = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : treePath();
==========
"use strict";

var buildPath = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : treePath();var _len5 = arguments.length,
    x = new Array(_len5 > 2 ? _len5 - 2 : 0),
    _key5 = 2;
==========
"use strict";

var _len5 = arguments.length,
    x = new Array(_len5 > 2 ? _len5 - 2 : 0),
    _key5 = 2;let flatLeaves = function () {
  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  return _fp.default.reject(next);
};
==========
"use strict";

var flatLeaves = function flatLeaves() {
  var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  return _fp.default.reject(next);
};let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
==========
"use strict";

var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;let tree = function () {
  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  let buildIteratee = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _fp.default.identity;
  let writeNode = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : writeProperty(next);
  return {
    walk: walk(next),
    walkAsync: walkAsync(next),
    transform: transformTree(next),
    reduce: reduceTree(next),
    toArrayBy: treeToArrayBy(next),
    toArray: treeToArray(next),
    leaves: leaves(next),
    leavesBy: leavesBy(next),
    lookup: treeLookup(next, buildIteratee),
    keyByWith: keyTreeByWith(next),
    traverse: next,
    flatten: flattenTree(next),
    flatLeaves: flatLeaves(next),
    map: mapTree(next, writeNode),
    mapLeaves: mapTreeLeaves(next, writeNode)
  };
};
==========
"use strict";

var tree = function tree() {
  var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
  var buildIteratee = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _fp.default.identity;
  var writeNode = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : writeProperty(next);
  return {
    walk: walk(next),
    walkAsync: walkAsync(next),
    transform: transformTree(next),
    reduce: reduceTree(next),
    toArrayBy: treeToArrayBy(next),
    toArray: treeToArray(next),
    leaves: leaves(next),
    leavesBy: leavesBy(next),
    lookup: treeLookup(next, buildIteratee),
    keyByWith: keyTreeByWith(next),
    traverse: next,
    flatten: flattenTree(next),
    flatLeaves: flatLeaves(next),
    map: mapTree(next, writeNode),
    mapLeaves: mapTreeLeaves(next, writeNode)
  };
};let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
==========
"use strict";

var next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;let buildIteratee = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _fp.default.identity;
==========
"use strict";

var buildIteratee = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _fp.default.identity;let writeNode = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : writeProperty(next);
==========
"use strict";

var writeNode = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : writeProperty(next);