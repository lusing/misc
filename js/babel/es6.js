var _fp = _interopRequireDefault(require("lodash/fp"));
var _function = require("./function");
var _collection = require("./collection");
var _conversion = require("./conversion");
// TODO: Move to proper files and exposelet callUnless = check => failFn => fn => (x, y) => check(x) ? failFn(y) : check(y) ? failFn(x) : fn(x, y);
let callUnlessEmpty = callUnless(_fp.default.isEmpty);
let wrapArray = x => [x];
let callUnlessEmptyArray = callUnlessEmpty(wrapArray);
let dropRight = _fp.default.dropRight(1);
let last = _fp.default.takeRight(1); // Arrays// ------
// Arrays// ------let compactJoin = _fp.default.curry((join, x) => _fp.default.compact(x).join(join));
let dotJoin = compactJoin('.');
let dotJoinWith = fn => x => _fp.default.filter(fn, x).join('.');
let repeated = _fp.default.flow(_fp.default.groupBy(e => e), _fp.default.filter(e => e.length > 1), _fp.default.flatten, _fp.default.uniq);
let push = _fp.default.curry((val, arr) => arr.concat([val]));
let pushIn = _fp.default.curry((arr, val) => arr.concat([val]));
let pushOn = _fp.default.curry((arr, val) => {  arr.push(val);  return arr;});
let moveIndex = (from, to, arr) => _fp.default.flow(_fp.default.pullAt(from), (0, _collection.insertAtIndex)(to, arr[from]))(arr);
let overlaps = (x, y) => y[0] > x[1];
let mergeRange = (x, y) => [[x[0], _fp.default.max(x.concat(y))]];
let actuallMergeRanges = callUnlessEmptyArray((x, y) => overlaps(x, y) ? [x, y] : mergeRange(x, y));
let mergeRanges = _fp.default.flow(_fp.default.sortBy([0, 1]), _fp.default.reduce((result, range) => dropRight(result).concat(actuallMergeRanges(_fp.default.flatten(last(result)), range)), [])); // [a, b...] -> a -> b
// [a, b...] -> a -> blet cycle = _fp.default.curry((a, n) => a[(a.indexOf(n) + 1) % a.length]);
let arrayToObject = _fp.default.curry((k, v, a) => _fp.default.flow(_fp.default.keyBy(k), _fp.default.mapValues(v))(a)); // zipObject that supports functions instead of objects
// zipObject that supports functions instead of objectslet zipObjectDeepWith = _fp.default.curry((x, y) => _fp.default.zipObjectDeep(x, _fp.default.isFunction(y) && _fp.default.isArray(x) ? _fp.default.times(y, x.length) : y));
let flags = zipObjectDeepWith(_fp.default, () => true);
let prefixes = list => _fp.default.range(1, list.length + 1).map(x => _fp.default.take(x, list));
let encoder = separator => ({  encode: compactJoin(separator),  decode: _fp.default.split(separator)});
let dotEncoder = encoder('.');
let slashEncoder = encoder('/');
let chunkBy = _fp.default.curry((f, array) => _fp.default.isEmpty(array) ? [] : _fp.default.reduce((acc, x) => f(_fp.default.last(acc), x) ? [..._fp.default.initial(acc), [..._fp.default.last(acc), x]] : [...acc, [x]], [[_fp.default.head(array)]], _fp.default.tail(array)));
let toggleElementBy = _fp.default.curry((check, val, arr) => ((0, _function.callOrReturn)(check, val, arr) ? _fp.default.pull : push)(val, arr));
let toggleElement = toggleElementBy(_fp.default.includes);
let intersperse = _fp.default.curry((f, _ref) => {  let [x0, ...xs] = _ref;  return (0, _conversion.reduceIndexed)((acc, x, i) => i === xs.length ? [...acc, x] : [...acc, (0, _function.callOrReturn)(f, acc, i, xs), x], [x0], xs);});
let [x0, ...xs] = _ref;
let replaceElementBy = _fp.default.curry((f, b, arr) => _fp.default.map(c => f(c) ? b : c, arr));
let replaceElement = _fp.default.curry((a, b, arr) => replaceElementBy(_fp.default.isEqual(a), b, arr));
var _fp = _interopRequireDefault(require("lodash/fp"));
var _tree = require("./tree");
const flowMap = function () {  return _fp.default.map(_fp.default.flow(...arguments));};
let findApply = _fp.default.curry((f, arr) => _fp.default.iteratee(f)(_fp.default.find(f, arr))); // Algebras// --------// A generic map that works for plain objects and arrays
// Algebras// --------// A generic map that works for plain objects and arrayslet map = _fp.default.curry((f, x) => (_fp.default.isArray(x) ? _fp.default.map : _fp.default.mapValues).convert({  cap: false})(f, x)); // Map for any recursive algebraic data structure// defaults in multidimensional arrays and recursive plain objects
// Map for any recursive algebraic data structure// defaults in multidimensional arrays and recursive plain objectslet deepMap = _fp.default.curry(function (fn, obj) {  let _map = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : map;  let is = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : _tree.isTraversable;  return _map(e => is(e) ? deepMap(fn, fn(e), _map, is) : e, obj);});
let _map = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : map;
let is = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : _tree.isTraversable;
let insertAtStringIndex = (index, val, str) => str.slice(0, index) + val + str.slice(index);
let insertAtArrayIndex = (index, val, arr) => {  let result = _fp.default.clone(arr);  result.splice(index, 0, val);  return result;};
let result = _fp.default.clone(arr);
let insertAtIndex = _fp.default.curry((index, val, collection) => _fp.default.isString(collection) ? insertAtStringIndex(index, val, collection) : insertAtArrayIndex(index, val, collection));
let compactMap = _fp.default.curry((fn, collection) => _fp.default.flow(_fp.default.map(fn), _fp.default.compact)(collection));
var _fp = _interopRequireDefault(require("lodash/fp"));
// (fn, a, b) -> fn(a, b)let maybeCall = function (fn) {  for (var _len = arguments.length, args = new Array(_len > 1 ? _len - 1 : 0), _key = 1; _key < _len; _key++) {    args[_key - 1] = arguments[_key];  }  return _fp.default.isFunction(fn) && fn(...args);}; // (fn, a, b) -> fn(a, b)
var _len = arguments.length,    args = new Array(_len > 1 ? _len - 1 : 0),    _key = 1;
// (fn, a, b) -> fn(a, b)let callOrReturn = function (fn) {  for (var _len2 = arguments.length, args = new Array(_len2 > 1 ? _len2 - 1 : 0), _key2 = 1; _key2 < _len2; _key2++) {    args[_key2 - 1] = arguments[_key2];  }  return _fp.default.isFunction(fn) ? fn(...args) : fn;}; // (a, Monoid f) -> f[a] :: f a
var _len2 = arguments.length,    args = new Array(_len2 > 1 ? _len2 - 1 : 0),    _key2 = 1;
// (a, Monoid f) -> f[a] :: f alet boundMethod = (method, object) => object[method].bind(object); // http://ramdajs.com/docs/#converge
// http://ramdajs.com/docs/#convergelet converge = (converger, branches) => function () {  return converger(_fp.default.over(branches)(...arguments));};
let composeApply = (f, g) => x => f(g(x))(x);
let comply = composeApply; // Prettier version of `defer` the one from bluebird docs
// Prettier version of `defer` the one from bluebird docslet defer = () => {  let resolve;  let reject;  let promise = new Promise((res, rej) => {    resolve = res;    reject = rej;  });  return {    resolve,    reject,    promise  };}; // `_.debounce` for async functions, which require consistently returning a single promise for all queued calls
let resolve;
let reject;
let promise = new Promise((res, rej) => {  resolve = res;  reject = rej;});
// `_.debounce` for async functions, which require consistently returning a single promise for all queued callslet debounceAsync = (n, f) => {  let deferred = defer();  let debounced = _fp.default.debounce(n, function () {    deferred.resolve(f(...arguments));    deferred = defer();  });  return function () {    debounced(...arguments);    return deferred.promise;  };};
let deferred = defer();
let debounced = _fp.default.debounce(n, function () {  deferred.resolve(f(...arguments));  deferred = defer();});
let currier = f => function () {  for (var _len3 = arguments.length, fns = new Array(_len3), _key3 = 0; _key3 < _len3; _key3++) {    fns[_key3] = arguments[_key3];  }  return _fp.default.curryN(fns[0].length, f(...fns));}; // (f1, f2, ...fn) -> f1Args1 -> f1Arg2 -> ...f1ArgN -> fn(f2(f1))
var _len3 = arguments.length,    fns = new Array(_len3),    _key3 = 0;
// (f1, f2, ...fn) -> f1Args1 -> f1Arg2 -> ...f1ArgN -> fn(f2(f1))let flurry = currier(_fp.default.flow); // like _.overArgs, but on all args
// like _.overArgs, but on all argslet mapArgs = _fp.default.curry((mapper, fn) => function () {  for (var _len4 = arguments.length, x = new Array(_len4), _key4 = 0; _key4 < _len4; _key4++) {    x[_key4] = arguments[_key4];  }  return fn(...x.map(mapper));});
var _len4 = arguments.length,    x = new Array(_len4),    _key4 = 0;
var _fp = _interopRequireDefault(require("lodash/fp"));
let differentLast = (normalCase, lastCase) => (acc, i, list) => i === list.length - 1 ? _fp.default.iteratee(lastCase)(acc, i, list) : _fp.default.iteratee(normalCase)(acc, i, list);
var _fp = _interopRequireDefault(require("lodash/fp"));
var _conversion = require("./conversion");
var _array = require("./array");
var _logic = require("./logic");
// Stubslet functionLens = val => function () {  if (!arguments.length) return val;  val = arguments.length <= 0 ? undefined : arguments[0];};
let objectLens = val => ({  get: () => val,  set(x) {    val = x;  }}); // Lens Conversion
// Lens Conversionlet fnToObj = fn => ({  get: fn,  set: fn});
let objToFn = lens => function () {  return arguments.length ? lens.set(arguments.length <= 0 ? undefined : arguments[0]) : lens.get();}; // Lens Construction
// Lens Constructionlet lensProp = (field, source) => ({  get: () => _fp.default.get(field, source),  //source[field],  set(value) {    (0, _conversion.setOn)(field, value, source); // source[field] = value  }}); // NOTE: This used to use mapValues; however, doing so would sometimes cause issues// in some edge cases like trying to lens state coming from an inject function// in the mobx library. It would inadvertently cause the inject to re-run.// Using reduce here alleviates that issue.
// NOTE: This used to use mapValues; however, doing so would sometimes cause issues// in some edge cases like trying to lens state coming from an inject function// in the mobx library. It would inadvertently cause the inject to re-run.// Using reduce here alleviates that issue.let lensOf = object => _fp.default.reduce((res, key) => {  res[key] = lensProp(key, object);  return res;}, {}, _fp.default.keys(object));
let includeLens = function (value) {  for (var _len = arguments.length, lens = new Array(_len > 1 ? _len - 1 : 0), _key = 1; _key < _len; _key++) {    lens[_key - 1] = arguments[_key];  }  return {    get: () => _fp.default.includes(value, view(...lens)),    // Uniq is to ensure multiple calls to set(true) don't push multiple times since this is about membership of a set    set: x => set(_fp.default.uniq((0, _array.toggleElementBy)(!x, value, view(...lens))), ...lens)  };}; // Lens Manipulation//let construct = (...lens) => (lens[1] ? lensProp(...lens) : lens[0])
var _len = arguments.length,    lens = new Array(_len > 1 ? _len - 1 : 0),    _key = 1;
// Lens Manipulation//let construct = (...lens) => (lens[1] ? lensProp(...lens) : lens[0])let lensPair = (get, set) => ({  get,  set});
let construct = function () {  for (var _len2 = arguments.length, args = new Array(_len2), _key2 = 0; _key2 < _len2; _key2++) {    args[_key2] = arguments[_key2];  }  return args[1] ? _fp.default.every(_fp.default.isFunction, args) ? lensPair(...args) : lensProp(...args) : (0, _logic.when)(_fp.default.isArray, stateLens)(args[0]);};
var _len2 = arguments.length,    args = new Array(_len2),    _key2 = 0;
let read = lens => lens.get ? lens.get() : lens();
let view = function () {  return read(construct(...arguments));};
let views = function () {  for (var _len3 = arguments.length, lens = new Array(_len3), _key3 = 0; _key3 < _len3; _key3++) {    lens[_key3] = arguments[_key3];  }  return () => view(...lens);};
var _len3 = arguments.length,    lens = new Array(_len3),    _key3 = 0;
let write = (val, lens) => lens.set ? lens.set(val) : lens(val);
let set = _fp.default.curryN(2, function (val) {  for (var _len4 = arguments.length, lens = new Array(_len4 > 1 ? _len4 - 1 : 0), _key4 = 1; _key4 < _len4; _key4++) {    lens[_key4 - 1] = arguments[_key4];  }  return write(val, construct(...lens));});
var _len4 = arguments.length,    lens = new Array(_len4 > 1 ? _len4 - 1 : 0),    _key4 = 1;
let sets = _fp.default.curryN(2, function (val) {  for (var _len5 = arguments.length, lens = new Array(_len5 > 1 ? _len5 - 1 : 0), _key5 = 1; _key5 < _len5; _key5++) {    lens[_key5 - 1] = arguments[_key5];  }  return () => set(val, ...lens);});
var _len5 = arguments.length,    lens = new Array(_len5 > 1 ? _len5 - 1 : 0),    _key5 = 1;
let setsWith = _fp.default.curry(function (f) {  for (var _len6 = arguments.length, lens = new Array(_len6 > 1 ? _len6 - 1 : 0), _key6 = 1; _key6 < _len6; _key6++) {    lens[_key6 - 1] = arguments[_key6];  }  return x => set(_fp.default.iteratee(f)(x), ...lens);});
var _len6 = arguments.length,    lens = new Array(_len6 > 1 ? _len6 - 1 : 0),    _key6 = 1;
let flip = function () {  for (var _len7 = arguments.length, lens = new Array(_len7), _key7 = 0; _key7 < _len7; _key7++) {    lens[_key7] = arguments[_key7];  }  return () => set(!view(...lens), ...lens);};
var _len7 = arguments.length,    lens = new Array(_len7),    _key7 = 0;
let on = sets(true);
let off = sets(false); // Lens Consumption// Map lens to dom event handlers
// Lens Consumption// Map lens to dom event handlerslet binding = (value, getEventValue) => function () {  for (var _len8 = arguments.length, lens = new Array(_len8), _key8 = 0; _key8 < _len8; _key8++) {    lens[_key8] = arguments[_key8];  }  return {    [value]: view(...lens),    onChange: setsWith(getEventValue, ...lens)  };}; // Dom events have relevent fields on the `target` property of event objects
var _len8 = arguments.length,    lens = new Array(_len8),    _key8 = 0;
// Dom events have relevent fields on the `target` property of event objectslet targetBinding = field => binding(field, (0, _logic.when)(_fp.default.hasIn(`target.${field}`), _fp.default.get(`target.${field}`)));
let domLens = {  value: targetBinding('value'),  checkboxValues: _fp.default.flow(includeLens, targetBinding('checked')),  hover: function () {    return {      onMouseEnter: on(...arguments),      onMouseLeave: off(...arguments)    };  },  focus: function () {    return {      onFocus: on(...arguments),      onBlur: off(...arguments)    };  },  targetBinding,  binding};
let stateLens = _ref => {  let [value, set] = _ref;  return {    get: () => value,    set  };};
let [value, set] = _ref;
var _fp = _interopRequireDefault(require("lodash/fp"));
var _array = require("./array");
var _logic = require("./logic");
var _lang = require("./lang");
var _conversion = require("./conversion");
var _collection = require("./collection");
var _aspect = require("./aspect");
var _function = require("./function");
const noCap = _fp.default.convert({  cap: false}); // (k, v) -> {k: v}
// (k, v) -> {k: v}const singleObject = _fp.default.curry((key, value) => ({  [key]: value}));
const singleObjectR = _fp.default.flip(singleObject); // Formerly objToObjArr// ({a, b}) -> [{a}, {b}]
// Formerly objToObjArr// ({a, b}) -> [{a}, {b}]const chunkObject = value => _fp.default.isArray(value) ? value : _fp.default.map(_fp.default.spread(singleObject), _fp.default.toPairs(value)); // Remove properties with falsey values: ({ a: 1, b: null, c: false}) -> {a:1}
// Remove properties with falsey values: ({ a: 1, b: null, c: false}) -> {a:1}const compactObject = _fp.default.pickBy(_fp.default.identity);
const isEmptyObject = _fp.default.isEqual({});
const isNotEmptyObject = _fp.default.negate(isEmptyObject); // { a:1, b:{}, c:2 } -> { a:1, c:2 }
// { a:1, b:{}, c:2 } -> { a:1, c:2 }const stripEmptyObjects = _fp.default.pickBy(isNotEmptyObject); // const crazyBS = (f, g) => (a, b) => f(a)(g(b))// { a: { b: 1, c: 2 } }, [ 'b' ] -> { a: { b: 1 } }
// const crazyBS = (f, g) => (a, b) => f(a)(g(b))// { a: { b: 1, c: 2 } }, [ 'b' ] -> { a: { b: 1 } }const pickInto = (map, source) => _fp.default.mapValues((0, _conversion.pickIn)(source), map);
const renameProperty = _fp.default.curry((from, to, target) => _fp.default.has(from, target) ? _fp.default.flow(x => _fp.default.set(to, _fp.default.get(from, x), x), _fp.default.unset(from))(target) : target); // { x:['a','b'], y:1 } -> [{ x:'a', y:1 }, { x:'b', y:1 }] just like mongo's `$unwind`
// { x:['a','b'], y:1 } -> [{ x:'a', y:1 }, { x:'b', y:1 }] just like mongo's `$unwind`const unwind = _fp.default.curry((prop, x) => (0, _logic.ifElse)(_fp.default.isArray, _fp.default.map(y => _fp.default.set(prop, y, x)), _fp.default.stubArray, _fp.default.get(prop, x))); // this one's _actually_ just like mongo's `$unwind`
// this one's _actually_ just like mongo's `$unwind`const unwindArray = _fp.default.curry((prop, xs) => _fp.default.flatMap(unwind(prop))(xs));
const isFlatObject = (0, _logic.overNone)([_fp.default.isPlainObject, _fp.default.isArray]); // { a: { b: { c: 1 } } } => { 'a.b.c' : 1 }
// { a: { b: { c: 1 } } } => { 'a.b.c' : 1 }const flattenObject = (input, paths) => (0, _conversion.reduceIndexed)((output, value, key) => _fp.default.merge(output, (isFlatObject(value) ? singleObjectR : flattenObject)(value, (0, _array.dotJoinWith)(_lang.isNotNil)([paths, key]))), {}, input); // { 'a.b.c' : 1 } => { a: { b: { c: 1 } } }
// { 'a.b.c' : 1 } => { a: { b: { c: 1 } } }const unflattenObject = x => _fp.default.zipObjectDeep(_fp.default.keys(x), _fp.default.values(x)); // Returns true if object keys are only elements from signature list (but does not require all signature keys to be present)
// Returns true if object keys are only elements from signature list (but does not require all signature keys to be present)const matchesSignature = _fp.default.curry((signature, value) => _fp.default.isObject(value) && !_fp.default.difference(_fp.default.keys(value), signature).length); // `_.matches` that returns true if one or more of the conditions match instead of all
// `_.matches` that returns true if one or more of the conditions match instead of allconst matchesSome = _fp.default.flow(chunkObject, _fp.default.map(_fp.default.matches), _fp.default.overSome); // Checks if a property deep in a given item equals to a given value
// Checks if a property deep in a given item equals to a given valueconst compareDeep = _fp.default.curry((path, item, value) => _fp.default.get(path, item) === value); //Depreacted in favor of _.update version from lodash
//Depreacted in favor of _.update version from lodashconst mapProp = _aspect.aspects.deprecate('mapProp', '1.46.0', '_.update')(noCap.update); // `_.get` that returns the target object if lookup fails
// `_.get` that returns the target object if lookup failslet getOrReturn = _fp.default.curry((prop, x) => _fp.default.getOr(x, prop, x)); // `_.get` that returns the prop if lookup fails
// `_.get` that returns the prop if lookup failslet alias = _fp.default.curry((prop, x) => _fp.default.getOr(prop, prop, x)); // flipped alias
// flipped aliaslet aliasIn = _fp.default.curry((x, prop) => _fp.default.getOr(prop, prop, x)); // A `_.get` that takes an array of paths and returns the value at the first path that matches
// A `_.get` that takes an array of paths and returns the value at the first path that matcheslet cascade = _fp.default.curryN(2, (paths, obj, defaultValue) => _fp.default.flow((0, _collection.findApply)(x => x && _fp.default.iteratee(x)(obj)), _fp.default.defaultTo(defaultValue))(paths)); // Flipped cascade
// Flipped cascadelet cascadeIn = _fp.default.curryN(2, (obj, paths, defaultValue) => cascade(paths, obj, defaultValue)); // A `_.get` that takes an array of paths and returns the first path that matched
// A `_.get` that takes an array of paths and returns the first path that matchedlet cascadeKey = _fp.default.curry((paths, obj) => _fp.default.find((0, _conversion.getIn)(obj), paths)); // A `_.get` that takes an array of paths and returns the first path that exists
// A `_.get` that takes an array of paths and returns the first path that existslet cascadePropKey = _fp.default.curry((paths, obj) => _fp.default.find((0, _conversion.hasIn)(obj), paths)); // A `_.get` that takes an array of paths and returns the first value that has an existing path
// A `_.get` that takes an array of paths and returns the first value that has an existing pathlet cascadeProp = _fp.default.curry((paths, obj) => _fp.default.get(cascadePropKey(paths, obj), obj));
let unkeyBy = _fp.default.curry((keyName, obj) => (0, _conversion.mapIndexed)((val, key) => _fp.default.extend(val, {  [keyName || key]: key}))(obj));
let simpleDiff = (original, deltas) => {  let o = flattenObject(original);  return _fp.default.flow(flattenObject, (0, _conversion.mapValuesIndexed)((to, field) => ({    from: o[field],    to  })), _fp.default.omitBy(x => _fp.default.isEqual(x.from, x.to)))(deltas);};
let o = flattenObject(original);
let simpleDiffArray = _fp.default.flow(simpleDiff, unkeyBy('field'));
let diff = (original, deltas) => {  let o = flattenObject(original);  let d = flattenObject(deltas);  return _fp.default.flow((0, _conversion.mapValuesIndexed)((_, field) => ({    from: o[field],    to: d[field]  })), _fp.default.omitBy(x => _fp.default.isEqual(x.from, x.to)))(_fp.default.merge(o, d));};
let o = flattenObject(original);
let d = flattenObject(deltas);
let diffArray = _fp.default.flow(diff, unkeyBy('field')); // A `_.pick` that mutates the object
// A `_.pick` that mutates the objectlet pickOn = function () {  let paths = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : [];  let obj = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};  return _fp.default.flow(_fp.default.keys, _fp.default.map(key => {    if (!_fp.default.includes(key, paths)) {      delete obj[key];    }  }))(obj);};
let paths = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : [];
let obj = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};
let mergeArrays = (objValue, srcValue) => _fp.default.isArray(objValue) ? objValue.concat(srcValue) : undefined; // Straight from the lodash docs
// Straight from the lodash docslet mergeAllArrays = _fp.default.mergeAllWith(mergeArrays); // { a: [x, y, z], b: [x] } -> { x: [a, b], y: [a], z: [a] }
// { a: [x, y, z], b: [x] } -> { x: [a, b], y: [a], z: [a] }let invertByArray = _fp.default.flow((0, _conversion.mapIndexed)((arr, key) => (0, _array.zipObjectDeepWith)(arr, () => [key])), mergeAllArrays); // key -> { a: { x: 1 }, b: { y: 2 } } -> { a: { x: 1, key: 'a' }, b: { y: 2, key: 'b' } }
// key -> { a: { x: 1 }, b: { y: 2 } } -> { a: { x: 1, key: 'a' }, b: { y: 2, key: 'b' } }const stampKey = _fp.default.curry((key, x) => (0, _conversion.mapValuesIndexed)((val, k) => ({ ...val,  [key]: k}), x));
let omitNil = x => _fp.default.omitBy(_fp.default.isNil, x);
let omitNull = x => _fp.default.omitBy(_fp.default.isNull, x);
let omitBlank = x => _fp.default.omitBy(_lang.isBlank, x);
let omitEmpty = x => _fp.default.omitBy(_fp.default.isEmpty, x); // ([f, g]) -> (x, y) -> {...f(x, y), ...g(x, y)}
// ([f, g]) -> (x, y) -> {...f(x, y), ...g(x, y)}let mergeOverAll = _fp.default.curryN(2, function (fns) {  for (var _len = arguments.length, x = new Array(_len > 1 ? _len - 1 : 0), _key = 1; _key < _len; _key++) {    x[_key - 1] = arguments[_key];  }  return _fp.default.flow(_fp.default.over(fns), _fp.default.mergeAll)(...x);}); // customizer -> ([f, g]) -> (x, y) -> {...f(x, y), ...g(x, y)}
var _len = arguments.length,    x = new Array(_len > 1 ? _len - 1 : 0),    _key = 1;
// customizer -> ([f, g]) -> (x, y) -> {...f(x, y), ...g(x, y)}let mergeOverAllWith = _fp.default.curryN(3, function (customizer, fns) {  for (var _len2 = arguments.length, x = new Array(_len2 > 2 ? _len2 - 2 : 0), _key2 = 2; _key2 < _len2; _key2++) {    x[_key2 - 2] = arguments[_key2];  }  return _fp.default.flow(_fp.default.over(fns), _fp.default.mergeAllWith(customizer))(...x);}); // ([f, g]) -> (x, y) -> {...f(x, y), ...g(x, y)}
var _len2 = arguments.length,    x = new Array(_len2 > 2 ? _len2 - 2 : 0),    _key2 = 2;
// ([f, g]) -> (x, y) -> {...f(x, y), ...g(x, y)}let mergeOverAllArrays = mergeOverAllWith(mergeArrays); // (x -> y) -> k -> {k: x} -> y
// (x -> y) -> k -> {k: x} -> ylet getWith = _fp.default.curry((customizer, path, object) => customizer(_fp.default.get(path, object))); // ({a} -> {b}) -> {a} -> {a, b}
// ({a} -> {b}) -> {a} -> {a, b}let expandObject = _fp.default.curry((transform, obj) => ({ ...obj,  ...transform(obj)})); // k -> (a -> {b}) -> {k: a} -> {a, b}
// k -> (a -> {b}) -> {k: a} -> {a, b}let expandObjectBy = _fp.default.curry((key, fn, obj) => expandObject(getWith(fn, key))(obj));
let commonKeys = _fp.default.curryN(2, (0, _function.mapArgs)(_fp.default.keys, _fp.default.intersection));
let findKeyIndexed = _fp.default.findKey.convert({  cap: false});
let firstCommonKey = _fp.default.curry((x, y) => findKeyIndexed((val, key) => _fp.default.has(key, x), y));
var _collection = require("./collection");
var _fp = _interopRequireDefault(require("lodash/fp"));
var _logic = require("./logic");
var _array = require("./array");
var _iterators = require("./iterators");
const wrap = (pre, post, content) => (pre || '') + content + (post || pre || '');
const quote = _fp.default.partial(wrap, ['"', '"']);
const parens = _fp.default.partial(wrap, ['(', ')']);
const concatStrings = _fp.default.flow(_fp.default.compact, _fp.default.map(_fp.default.trim), _fp.default.join(' '));
const trimStrings = (0, _collection.map)((0, _logic.when)(_fp.default.isString, _fp.default.trim)); // _.startCase does the trick, deprecate it!
// _.startCase does the trick, deprecate it!let autoLabel = _fp.default.startCase;
let autoLabelOption = a => ({  value: (0, _logic.when)(_fp.default.isUndefined, a)(a.value),  label: a.label || autoLabel((0, _logic.when)(_fp.default.isUndefined, a)(a.value))});
let autoLabelOptions = _fp.default.map(autoLabelOption);
let toSentenceWith = _fp.default.curry((separator, lastSeparator, array) => _fp.default.flow((0, _array.intersperse)((0, _iterators.differentLast)(() => separator, () => lastSeparator)), _fp.default.join(''))(array));
let toSentence = toSentenceWith(', ', ' and '); // ((array -> object), array) -> string -> string
// ((array -> object), array) -> string -> stringlet uniqueStringWith = _fp.default.curry((cachizer, initialKeys) => {  let f = x => {    let result = x;    while (cache[result]) {      result = x + cache[x];      cache[x] += 1;    }    cache[result] = (cache[result] || 0) + 1;    return result;  };  let cache = cachizer(initialKeys);  f.cache = cache;  f.clear = () => {    cache = {};    f.cache = cache;  };  return f;});
let f = x => {  let result = x;  while (cache[result]) {    result = x + cache[x];    cache[x] += 1;  }  cache[result] = (cache[result] || 0) + 1;  return result;};
let result = x;
let cache = cachizer(initialKeys);
let uniqueString = function () {  let arr = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : [];  return uniqueStringWith(_fp.default.countBy(_fp.default.identity), arr);};
let arr = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : [];
var _fp = _interopRequireDefault(require("lodash/fp"));
var _conversion = require("./conversion");
var _lang = require("./lang");
// Corelet aspect = _ref => {  let {    name = 'aspect',    init = _fp.default.noop,    after = _fp.default.noop,    before = _fp.default.noop,    always = _fp.default.noop,    onError = _lang.throws // ?: interceptParams, interceptResult, wrap  } = _ref;  return f => {    let {      state = {}    } = f;    init(state); // Trick to set function.name of anonymous function    let x = {      [name]() {        for (var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++) {          args[_key] = arguments[_key];        }        let result;        let error;        return Promise.resolve().then(() => before(args, state)).then(() => f(...args)).then(r => {          result = r;        }).then(() => after(result, state, args)).catch(e => onError(e, state, args)).catch(e => {          error = e;        }).then(() => always(state, args)).then(() => {          if (error) throw error;        }).then(() => result);      }    };    x[name].state = state;    return x[name];  };};
let {  name = 'aspect',  init = _fp.default.noop,  after = _fp.default.noop,  before = _fp.default.noop,  always = _fp.default.noop,  onError = _lang.throws // ?: interceptParams, interceptResult, wrap} = _ref;
let {  state = {}} = f;
// Trick to set function.name of anonymous functionlet x = {  [name]() {    for (var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++) {      args[_key] = arguments[_key];    }    let result;    let error;    return Promise.resolve().then(() => before(args, state)).then(() => f(...args)).then(r => {      result = r;    }).then(() => after(result, state, args)).catch(e => onError(e, state, args)).catch(e => {      error = e;    }).then(() => always(state, args)).then(() => {      if (error) throw error;    }).then(() => result);  }};
var _len = arguments.length,    args = new Array(_len),    _key = 0;
let result;
let error;
let aspectSync = _ref2 => {  let {    name = 'aspect',    init = _fp.default.noop,    after = _fp.default.noop,    before = _fp.default.noop,    always = _fp.default.noop,    onError = _lang.throws // ?: interceptParams, interceptResult, wrap  } = _ref2;  return f => {    let {      state = {}    } = f;    init(state); // Trick to set function.name of anonymous function    let x = {      [name]() {        for (var _len2 = arguments.length, args = new Array(_len2), _key2 = 0; _key2 < _len2; _key2++) {          args[_key2] = arguments[_key2];        }        try {          before(args, state);          let result = f(...args);          after(result, state, args);          return result;        } catch (e) {          onError(e, state, args);          throw e;        } finally {          always(state, args);        }      }    };    x[name].state = state;    return x[name];  };}; // Example Aspects
let {  name = 'aspect',  init = _fp.default.noop,  after = _fp.default.noop,  before = _fp.default.noop,  always = _fp.default.noop,  onError = _lang.throws // ?: interceptParams, interceptResult, wrap} = _ref2;
let {  state = {}} = f;
// Trick to set function.name of anonymous functionlet x = {  [name]() {    for (var _len2 = arguments.length, args = new Array(_len2), _key2 = 0; _key2 < _len2; _key2++) {      args[_key2] = arguments[_key2];    }    try {      before(args, state);      let result = f(...args);      after(result, state, args);      return result;    } catch (e) {      onError(e, state, args);      throw e;    } finally {      always(state, args);    }  }};
var _len2 = arguments.length,    args = new Array(_len2),    _key2 = 0;
let result = f(...args);
// Example Aspectslet logs = function () {  let extend = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : _conversion.defaultsOn;  return aspect({    init: extend({      logs: []    }),    after: (result, state) => state.logs.push(result),    name: 'logs'  });};
let extend = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : _conversion.defaultsOn;
let error = function () {  let extend = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : _conversion.defaultsOn;  return aspect({    init: extend({      error: null    }),    onError: (0, _conversion.setOn)('error'),    name: 'error'  });};
let extend = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : _conversion.defaultsOn;
let errors = function () {  let extend = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : _conversion.defaultsOn;  return aspect({    init: extend({      errors: []    }),    onError: (e, state) => state.errors.push(e),    name: 'errors'  });};
let extend = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : _conversion.defaultsOn;
let status = function () {  let extend = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : _conversion.defaultsOn;  return aspect({    init: extend({      status: null,      processing: false,      succeeded: false,      failed: false,      // Computed get/set properties don't work, probably because lodash extend methods don't support copying them      setStatus(x) {        this.status = x;        this.failed = x === 'failed';        this.succeeded = x === 'succeeded';        this.processing = x === 'processing';      }    }),    before(params, state) {      state.setStatus('processing');    },    after(result, state) {      state.setStatus('succeeded');    },    onError: (0, _lang.tapError)((e, state) => {      state.setStatus('failed');    }),    name: 'status'  });};
let extend = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : _conversion.defaultsOn;
let clearStatus = function () {  let timeout = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : 500;  return aspect({    always(state) {      if (timeout !== null) {        setTimeout(() => {          state.setStatus(null);        }, timeout);      }    },    name: 'clearStatus'  });}; // This is a function just for consistency
let timeout = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : 500;
// This is a function just for consistencylet concurrency = () => aspect({  before(params, state) {    if (state.processing) {      throw Error('Concurrent Runs Not Allowed');    }  },  name: 'concurrency'});
let command = (extend, timeout) => _fp.default.flow(status(extend), clearStatus(timeout), concurrency(extend), error(extend));
let deprecate = (subject, version, alternative) => aspectSync({  before: () => console.warn(`\`${subject}\` is deprecated${version ? ` as of ${version}` : ''}${alternative ? ` in favor of \`${alternative}\`` : ''} ${_fp.default.trim((Error().stack || '').split('\n')[3])}`)});
let aspects = {  logs,  error,  errors,  status,  deprecate,  clearStatus,  concurrency,  command};
var _fp = _interopRequireDefault(require("lodash/fp"));
var _aspect = require("./aspect");
const noRearg = _fp.default.convert({  rearg: false});
const mutable = _fp.default.convert({  immutable: false});
const noCap = _fp.default.convert({  cap: false}); // Flips// ----------
// Flips// ----------const getIn = noRearg.get;
const hasIn = noRearg.has;
const pickIn = noRearg.pick;
const includesIn = noRearg.includes;
const inversions = _fp.default.mapKeys(k => `${k}In`, noRearg); // Mutables// ----------
// Mutables// ----------const extendOn = mutable.extend;
const defaultsOn = mutable.defaults;
const mergeOn = mutable.merge;
const setOn = mutable.set; // Curry required until https://github.com/lodash/lodash/issues/3440 is resolved
// Curry required until https://github.com/lodash/lodash/issues/3440 is resolvedconst unsetOn = _fp.default.curryN(2, mutable.unset);
const pullOn = mutable.pull;
const updateOn = mutable.update; // Uncaps// ------// Un-prefixed Deprecated
// Uncaps// ------// Un-prefixed Deprecatedconst reduce = _aspect.aspects.deprecate('reduce', '1.28.0', 'reduceIndexed')(noCap.reduce);
const mapValues = _aspect.aspects.deprecate('mapValues', '1.28.0', 'mapValuesIndexed')(noCap.mapValues);
const each = _aspect.aspects.deprecate('each', '1.28.0', 'eachIndexed')(noCap.each);
const mapIndexed = noCap.map;
const findIndexed = noCap.find;
const eachIndexed = noCap.each;
const reduceIndexed = noCap.reduce;
const pickByIndexed = noCap.pickBy;
const mapValuesIndexed = noCap.mapValues;
var _fp = _interopRequireDefault(require("lodash/fp"));
var _tree = require("./tree");
let throws = x => {  throw x;};
let tapError = f => function (e) {  for (var _len = arguments.length, args = new Array(_len > 1 ? _len - 1 : 0), _key = 1; _key < _len; _key++) {    args[_key - 1] = arguments[_key];  }  f(e, ...args);  throw e;};
var _len = arguments.length,    args = new Array(_len > 1 ? _len - 1 : 0),    _key = 1;
let isNotNil = _fp.default.negate(_fp.default.isNil);
let exists = isNotNil;
let isMultiple = x => (x || []).length > 1;
let append = _fp.default.curry((x, y) => y + x); // True for everything except null, undefined, '', [], and {}
// True for everything except null, undefined, '', [], and {}let isBlank = _fp.default.overSome([_fp.default.isNil, _fp.default.isEqual(''), _fp.default.isEqual([]), _fp.default.isEqual({})]);
let isNotBlank = _fp.default.negate(isBlank);
let isBlankDeep = combinator => x => combinator(isBlank, (0, _tree.tree)().leaves(x));
var _fp = _interopRequireDefault(require("lodash/fp"));
var _function = require("./function");
var _lang = require("./lang");
// ([f, g]) -> !f(x) && !g(x)const overNone = _fp.default.flow(_fp.default.overSome, _fp.default.negate);
let boolIteratee = x => _fp.default.isBoolean(x) || _fp.default.isNil(x) ? () => x : _fp.default.iteratee(x); // Port from Ramda
// Port from Ramdalet ifElse = _fp.default.curry((condition, onTrue, onFalse, x) => boolIteratee(condition)(x) ? (0, _function.callOrReturn)(onTrue, x) : (0, _function.callOrReturn)(onFalse, x));
let when = _fp.default.curry((condition, t, x) => ifElse(condition, t, _fp.default.identity, x));
let unless = _fp.default.curry((condition, f, x) => ifElse(condition, _fp.default.identity, f, x));
let whenExists = when(_lang.exists);
let whenTruthy = when(Boolean);
var _fp = _interopRequireDefault(require("lodash/fp"));
var _array = require("./array");
var _collection = require("./collection");
const testRegex = _fp.default.curry((regex, str) => new RegExp(regex).test(str));
const makeRegex = options => text => RegExp(text, options);
const makeAndTest = options => _fp.default.flow(makeRegex(options), testRegex);
const anyWordToRegexp = _fp.default.flow(_fp.default.words, _fp.default.join('|'));
const wordsToRegexp = _fp.default.flow(_fp.default.words, _fp.default.map(x => `(?=.*${x}.*)`), _fp.default.join(''), x => `.*${x}.*`);
const matchWords = _fp.default.curry((buildRegex, x) => {  // Not inlining so that we don't create the regexp every time  const regexp = RegExp(buildRegex(x), 'gi');  return y => !!(y && y.match(regexp));});
// Not inlining so that we don't create the regexp every timeconst regexp = RegExp(buildRegex(x), 'gi');
const matchAllWords = matchWords(wordsToRegexp);
const matchAnyWord = matchWords(anyWordToRegexp);
const allMatches = _fp.default.curry((regexStr, str) => {  let matched;  const regex = new RegExp(regexStr, 'g');  const result = [];  while ((matched = regex.exec(str)) !== null) {    result.push({      text: matched[0],      start: matched.index,      end: regex.lastIndex    });  }  return result;});
let matched;
const regex = new RegExp(regexStr, 'g');
const result = [];
const postings = _fp.default.curry((regex, str) => {  var match = regex.exec(str);  let result = [];  if (regex.flags.indexOf('g') < 0 && match) {    result.push([match.index, match.index + match[0].length]);  } else {    while (match) {      result.push([match.index, regex.lastIndex]);      match = regex.exec(str);    }  }  return result;});
var match = regex.exec(str);
let result = [];
const postingsForWords = _fp.default.curry((string, str) => _fp.default.reduce((result, word) => (0, _array.push)(postings(RegExp(word, 'gi'), str), result), [])(_fp.default.words(string)));
const highlightFromPostings = _fp.default.curry((start, end, postings, str) => {  let offset = 0;  _fp.default.each(posting => {    str = (0, _collection.insertAtIndex)(posting[0] + offset, start, str);    offset += start.length;    str = (0, _collection.insertAtIndex)(posting[1] + offset, end, str);    offset += end.length;  }, (0, _array.mergeRanges)(postings));  return str;});
let offset = 0;
const highlight = _fp.default.curry((start, end, pattern, input) => highlightFromPostings(start, end, _fp.default.isRegExp(pattern) ? postings(pattern, input) : _fp.default.flatten(postingsForWords(pattern, input)), input));
var _fp = _interopRequireDefault(require("lodash/fp"));
var _conversion = require("./conversion");
var _array = require("./array");
let isTraversable = x => _fp.default.isArray(x) || _fp.default.isPlainObject(x);
let traverse = x => isTraversable(x) && !_fp.default.isEmpty(x) && x;
let walk = function () {  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;  return function (pre) {    let post = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _fp.default.noop;    let parents = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : [];    let parentIndexes = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : [];    return (tree, index) => pre(tree, index, parents, parentIndexes) || (0, _conversion.findIndexed)(walk(next)(pre, post, [tree, ...parents], [index, ...parentIndexes]), next(tree, index, parents, parentIndexes) || []) || post(tree, index, parents, parentIndexes);  };}; // async/await is so much cleaner but causes regeneratorRuntime shenanigans// export let findIndexedAsync = async (f, data) => {//   for (let key in data) {//     if (await f(data[key], key, data)) return data[key]//   }// }// The general idea here is to keep popping off key/value pairs until we hit something that matches
let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
let post = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _fp.default.noop;
let parents = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : [];
let parentIndexes = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : [];
// async/await is so much cleaner but causes regeneratorRuntime shenanigans// export let findIndexedAsync = async (f, data) => {//   for (let key in data) {//     if (await f(data[key], key, data)) return data[key]//   }// }// The general idea here is to keep popping off key/value pairs until we hit something that matcheslet findIndexedAsync = function (f, data) {  let remaining = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : _fp.default.toPairs(data);  if (!remaining.length) return;  let [[key, val], ...rest] = remaining;  return Promise.resolve(f(val, key, data)).then(result => result ? val : rest.length ? findIndexedAsync(f, data, rest) : undefined);};
let remaining = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : _fp.default.toPairs(data);
let [[key, val], ...rest] = remaining;
let walkAsync = function () {  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;  return function (pre) {    let post = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _fp.default.noop;    let parents = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : [];    let parentIndexes = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : [];    return (tree, index) => Promise.resolve(pre(tree, index, parents, parentIndexes)).then(preResult => preResult || findIndexedAsync(walkAsync(next)(pre, post, [tree, ...parents], [index, ...parentIndexes]), next(tree, index, parents, parentIndexes) || [])).then(stepResult => stepResult || post(tree, index, parents, parentIndexes));  };};
let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
let post = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _fp.default.noop;
let parents = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : [];
let parentIndexes = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : [];
let transformTree = function () {  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;  return _fp.default.curry((f, x) => {    let result = _fp.default.cloneDeep(x);    walk(next)(f)(result);    return result;  });};
let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
let result = _fp.default.cloneDeep(x);
let reduceTree = function () {  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;  return _fp.default.curry((f, result, tree) => {    walk(next)(function () {      for (var _len = arguments.length, x = new Array(_len), _key = 0; _key < _len; _key++) {        x[_key] = arguments[_key];      }      result = f(result, ...x);    })(tree);    return result;  });};
let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
var _len = arguments.length,    x = new Array(_len),    _key = 0;
let writeProperty = function () {  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;  return (node, index, _ref) => {    let [parent] = _ref;    next(parent)[index] = node;  };};
let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
let [parent] = _ref;
let mapTree = function () {  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;  let writeNode = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : writeProperty(next);  return _fp.default.curry((mapper, tree) => transformTree(next)(function (node, i, parents) {    for (var _len2 = arguments.length, args = new Array(_len2 > 3 ? _len2 - 3 : 0), _key2 = 3; _key2 < _len2; _key2++) {      args[_key2 - 3] = arguments[_key2];    }    if (parents.length) writeNode(mapper(node, i, parents, ...args), i, parents, ...args);  })(mapper(tree)) // run mapper on root, and skip root in traversal  );};
let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
let writeNode = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : writeProperty(next);
var _len2 = arguments.length,    args = new Array(_len2 > 3 ? _len2 - 3 : 0),    _key2 = 3;
let mapTreeLeaves = function () {  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;  let writeNode = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : writeProperty(next);  return _fp.default.curry((mapper, tree) => // this unless wrapping can be done in user land, this is pure convenience  // mapTree(next, writeNode)(F.unless(next, mapper), tree)  mapTree(next, writeNode)(node => next(node) ? node : mapper(node), tree));};
let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
let writeNode = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : writeProperty(next);
let treeToArrayBy = function () {  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;  return _fp.default.curry((fn, tree) => reduceTree(next)(function (r) {    for (var _len3 = arguments.length, args = new Array(_len3 > 1 ? _len3 - 1 : 0), _key3 = 1; _key3 < _len3; _key3++) {      args[_key3 - 1] = arguments[_key3];    }    return (0, _array.push)(fn(...args), r);  }, [], tree));};
let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
var _len3 = arguments.length,    args = new Array(_len3 > 1 ? _len3 - 1 : 0),    _key3 = 1;
let treeToArray = function () {  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;  return treeToArrayBy(next)(x => x);}; // This could reuse treeToArrayBy and just reject traversable elements after, but this is more efficient// We can potentially unify these with tree transducers
let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
// This could reuse treeToArrayBy and just reject traversable elements after, but this is more efficient// We can potentially unify these with tree transducerslet leavesBy = function () {  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;  return _fp.default.curry((fn, tree) => reduceTree(next)(function (r, node) {    for (var _len4 = arguments.length, args = new Array(_len4 > 2 ? _len4 - 2 : 0), _key4 = 2; _key4 < _len4; _key4++) {      args[_key4 - 2] = arguments[_key4];    }    return next(node) ? r : (0, _array.push)(fn(node, ...args), r);  }, [], tree));};
let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
var _len4 = arguments.length,    args = new Array(_len4 > 2 ? _len4 - 2 : 0),    _key4 = 2;
let leaves = function () {  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;  return leavesBy(next)(x => x);};
let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
let treeLookup = function () {  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;  let buildIteratee = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _fp.default.identity;  return _fp.default.curry((path, tree) => _fp.default.reduce((tree, path) => (0, _conversion.findIndexed)(buildIteratee(path), next(tree)), tree, path));};
let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
let buildIteratee = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _fp.default.identity;
let keyTreeByWith = function () {  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;  return _fp.default.curry((transformer, groupIteratee, x) => _fp.default.flow(treeToArrayBy(next)(_fp.default.iteratee(groupIteratee)), _fp.default.uniq, _fp.default.keyBy(_fp.default.identity), _fp.default.mapValues(group => transformTree(next)(node => {    let matches = _fp.default.iteratee(groupIteratee)(node) === group;    transformer(node, matches, group);  }, x)))(x));}; // Flat Tree
let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
let matches = _fp.default.iteratee(groupIteratee)(node) === group;
// Flat Treelet treeKeys = (x, i, xs, is) => [i, ...is];
let treeValues = (x, i, xs) => [x, ...xs];
let treePath = function () {  let build = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : treeKeys;  let encoder = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _array.dotEncoder;  return function () {    return (encoder.encode || encoder)(build(...arguments).reverse());  };};
let build = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : treeKeys;
let encoder = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _array.dotEncoder;
let propTreePath = prop => treePath(_fp.default.flow(treeValues, _fp.default.map(prop)), _array.slashEncoder);
let flattenTree = function () {  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;  return function () {    let buildPath = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : treePath();    return reduceTree(next)(function (result, node) {      for (var _len5 = arguments.length, x = new Array(_len5 > 2 ? _len5 - 2 : 0), _key5 = 2; _key5 < _len5; _key5++) {        x[_key5 - 2] = arguments[_key5];      }      return _fp.default.set([buildPath(node, ...x)], node, result);    }, {});  };};
let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
let buildPath = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : treePath();
var _len5 = arguments.length,    x = new Array(_len5 > 2 ? _len5 - 2 : 0),    _key5 = 2;
let flatLeaves = function () {  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;  return _fp.default.reject(next);};
let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
let tree = function () {  let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;  let buildIteratee = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _fp.default.identity;  let writeNode = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : writeProperty(next);  return {    walk: walk(next),    walkAsync: walkAsync(next),    transform: transformTree(next),    reduce: reduceTree(next),    toArrayBy: treeToArrayBy(next),    toArray: treeToArray(next),    leaves: leaves(next),    leavesBy: leavesBy(next),    lookup: treeLookup(next, buildIteratee),    keyByWith: keyTreeByWith(next),    traverse: next,    flatten: flattenTree(next),    flatLeaves: flatLeaves(next),    map: mapTree(next, writeNode),    mapLeaves: mapTreeLeaves(next, writeNode)  };};
let next = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : traverse;
let buildIteratee = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _fp.default.identity;
let writeNode = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : writeProperty(next);
