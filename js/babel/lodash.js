const babel = require("@babel/core");

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
    console.log(str2);

    //let str0 = code.replace(/[\n\t]/g,'');
    //console.log(str0);

    console.log('------------------');
}

const codes = [
    `
function map(array, iteratee) {
    let index = -1
    const length = array == null ? 0 : array.length
    const result = new Array(length)
  
    while (++index < length) {
      result[index] = iteratee(array[index], index, array)
    }
    return result
  }
`,
    `
const add = createMathOperation((augend, addend) => augend + addend, 0)
`,
    `
function after(n, func) {
    if (typeof func !== 'function') {
      throw new TypeError('Expected a function')
    }
    n = n || 0
    return function(...args) {
      if (--n < 1) {
        return func.apply(this, args)
      }
    }
  }
`,
    `const at = (object, ...paths) => baseAt(object, baseFlatten(paths, 1))
`,
    `function attempt(func, ...args) {
    try {
      return func(...args)
    } catch (e) {
      return isError(e) ? e : new Error(e)
    }
  }
`,
    `function before(n, func) {
    let result
    if (typeof func !== 'function') {
      throw new TypeError('Expected a function')
    }
    return function(...args) {
      if (--n > 0) {
        result = func.apply(this, args)
      }
      if (n <= 1) {
        func = undefined
      }
      return result
    }
  }`,
    `const camelCase = (string) => (
    words(toString(string).replace(/['\u2019]/g, '')).reduce((result, word, index) => {
      word = word.toLowerCase()
      return result + (index ? upperFirst(word) : word)
    }, '')
  )`,
    `const camelCase = (string) => (
    words(toString(string).replace(/['\u2019]/g, '')).reduce((result, word, index) => {
      word = word.toLowerCase()
      return result + (index ? upperFirst(word) : word)
    }, '')
  )`,
    `const capitalize = (string) => upperFirst(toString(string).toLowerCase())`,
    `function castArray(...args) {
    if (!args.length) {
      return []
    }
    const value = args[0]
    return Array.isArray(value) ? value : [value]
  }`,
    `const ceil = createRound('ceil')`,
    `function chunk(array, size = 1) {
    size = Math.max(toInteger(size), 0)
    const length = array == null ? 0 : array.length
    if (!length || size < 1) {
      return []
    }
    let index = 0
    let resIndex = 0
    const result = new Array(Math.ceil(length / size))
  
    while (index < length) {
      result[resIndex++] = slice(array, index, (index += size))
    }
    return result
  }`,
    `function clamp(number, lower, upper) {
    number = +number
    lower = +lower
    upper = +upper
    lower = lower === lower ? lower : 0
    upper = upper === upper ? upper : 0
    if (number === number) {
      number = number <= upper ? number : upper
      number = number >= lower ? number : lower
    }
    return number
  }`,
    `function clone(value) {
    return baseClone(value, CLONE_SYMBOLS_FLAG)
  }`,
    `function cloneDeep(value) {
    return baseClone(value, CLONE_DEEP_FLAG | CLONE_SYMBOLS_FLAG)
  }`,
    `function cloneDeepWith(value, customizer) {
    customizer = typeof customizer === 'function' ? customizer : undefined
    return baseClone(value, CLONE_DEEP_FLAG | CLONE_SYMBOLS_FLAG, customizer)
  }`,
    `function cloneWith(value, customizer) {
    customizer = typeof customizer === 'function' ? customizer : undefined
    return baseClone(value, CLONE_SYMBOLS_FLAG, customizer)
  }
  `,
    `function compact(array) {
    let resIndex = 0
    const result = []
  
    if (array == null) {
      return result
    }
  
    for (const value of array) {
      if (value) {
        result[resIndex++] = value
      }
    }
    return result
  }`,
  `function cond(pairs) {
    const length = pairs == null ? 0 : pairs.length
  
    pairs = !length ? [] : map(pairs, (pair) => {
      if (typeof pair[1] !== 'function') {
        throw new TypeError('Expected a function')
      }
      return [pair[0], pair[1]]
    })
  
    return (...args) => {
      for (const pair of pairs) {
        if (pair[0].apply(this, args)) {
          return pair[1].apply(this, args)
        }
      }
    }
  }`,
  `function conforms(source) {
    return baseConforms(baseClone(source, CLONE_DEEP_FLAG))
  }`,
  `function conformsTo(object, source) {
    return source == null || baseConformsTo(object, source, keys(source))
  }`,
  `function countBy(collection, iteratee) {
    return reduce(collection, (result, value, key) => {
      key = iteratee(value)
      if (hasOwnProperty.call(result, key)) {
        ++result[key]
      } else {
        baseAssignValue(result, key, 1)
      }
      return result
    }, {})
  }`,
  `function create(prototype, properties) {
    prototype = prototype === null ? null : Object(prototype)
    const result = Object.create(prototype)
    return properties == null ? result : Object.assign(result, properties)
  }`,
  `  function invokeFunc(time) {
    const args = lastArgs
    const thisArg = lastThis

    lastArgs = lastThis = undefined
    lastInvokeTime = time
    result = func.apply(thisArg, args)
    return result
  }`,
  `  function startTimer(pendingFunc, wait) {
    if (useRAF) {
      root.cancelAnimationFrame(timerId)
      return root.requestAnimationFrame(pendingFunc)
    }
    return setTimeout(pendingFunc, wait)
  }`,
  `  function cancelTimer(id) {
    if (useRAF) {
      return root.cancelAnimationFrame(id)
    }
    clearTimeout(id)
  }`,
  `  function remainingWait(time) {
    const timeSinceLastCall = time - lastCallTime
    const timeSinceLastInvoke = time - lastInvokeTime
    const timeWaiting = wait - timeSinceLastCall

    return maxing
      ? Math.min(timeWaiting, maxWait - timeSinceLastInvoke)
      : timeWaiting
  }`,
  `function deburr(string) {
    return string && string.replace(reLatin, deburrLetter).replace(reComboMark, '')
  }`,
  `function defaultTo(value, defaultValue) {
    return (value == null || value !== value) ? defaultValue : value
  }
  `,
  `function defaultToAny(value, ...defaultValues) {
    return arrayReduce(defaultValues, defaultTo, value)
  }`,
  `function defaults(object, ...sources) {
    object = Object(object)
    sources.forEach((source) => {
      if (source != null) {
        source = Object(source)
        for (const key in source) {
          const value = object[key]
          if (value === undefined ||
              (eq(value, objectProto[key]) && !hasOwnProperty.call(object, key))) {
            object[key] = source[key]
          }
        }
      }
    })
    return object
  }`,
  `function defaultsDeep(...args) {
    args.push(undefined, customDefaultsMerge)
    return mergeWith.apply(undefined, args)
  }`,
  `function defer(func, ...args) {
    if (typeof func !== 'function') {
      throw new TypeError('Expected a function')
    }
    return setTimeout(func, 1, ...args)
  }`,
  `function delay(func, wait, ...args) {
    if (typeof func !== 'function') {
      throw new TypeError('Expected a function')
    }
    return setTimeout(func, +wait || 0, ...args)
  }`,
  `function difference(array, ...values) {
    return isArrayLikeObject(array)
      ? baseDifference(array, baseFlatten(values, 1, isArrayLikeObject, true))
      : []
  }`,
  `function differenceBy(array, ...values) {
    let iteratee = last(values)
    if (isArrayLikeObject(iteratee)) {
      iteratee = undefined
    }
    return isArrayLikeObject(array)
      ? baseDifference(array, baseFlatten(values, 1, isArrayLikeObject, true), iteratee)
      : []
  }`,
  `function differenceWith(array, ...values) {
    let comparator = last(values)
    if (isArrayLikeObject(comparator)) {
      comparator = undefined
    }
    return isArrayLikeObject(array)
      ? baseDifference(array, baseFlatten(values, 1, isArrayLikeObject, true), undefined, comparator)
      : []
  }`,
  `const divide = createMathOperation((dividend, divisor) => dividend / divisor, 1)`,
  `function drop(array, n=1) {
    const length = array == null ? 0 : array.length
    return length
      ? slice(array, n < 0 ? 0 : toInteger(n), length)
      : []
  }`,
  `function dropRight(array, n=1) {
    const length = array == null ? 0 : array.length
    n = length - toInteger(n)
    return length ? slice(array, 0, n < 0 ? 0 : n) : []
  }`,
  `function dropRightWhile(array, predicate) {
    return (array != null && array.length)
      ? baseWhile(array, predicate, true, true)
      : []
  }`,
  `function endsWith(string, target, position) {
    const { length } = string
    position = position === undefined ? length : +position
    if (position < 0 || position != position) {
      position = 0
    }
    else if (position > length) {
      position = length
    }
    const end = position
    position -= target.length
    return position >= 0 && string.slice(position, end) == target
  }`,
  `function eq(value, other) {
    return value === other || (value !== value && other !== other)
  }`,
  `function isEqual(value, other) {
    return baseIsEqual(value, other)
  }`,
  `function escape(string) {
    return (string && reHasUnescapedHtml.test(string))
      ? string.replace(reUnescapedHtml, (chr) => htmlEscapes[chr])
      : (string || '')
  }`,
  `function escapeRegExp(string) {
    return (string && reHasRegExpChar.test(string))
      ? string.replace(reRegExpChar, '\\$&')
      : (string || '')
  }`,
  `function every(array, predicate) {
    let index = -1
    const length = array == null ? 0 : array.length
  
    while (++index < length) {
      if (!predicate(array[index], index, array)) {
        return false
      }
    }
    return true
  }`,
  `function everyValue(object, predicate) {
    object = Object(object)
    const props = Object.keys(object)
  
    for (const key of props) {
      if (!predicate(object[key], key, object)) {
        return false
      }
    }
    return true
  }`,
  `function filter(array, predicate) {
    let index = -1
    let resIndex = 0
    const length = array == null ? 0 : array.length
    const result = []
  
    while (++index < length) {
      const value = array[index]
      if (predicate(value, index, array)) {
        result[resIndex++] = value
      }
    }
    return result
  }`,
  `function filterObject(object, predicate) {
    object = Object(object)
    const result = []
  
    Object.keys(object).forEach((key) => {
      const value = object[key]
      if (predicate(value, key, object)) {
        result.push(value)
      }
    })
    return result
  }`,
  `function findKey(object, predicate) {
    let result
    if (object == null) {
      return result
    }
    Object.keys(object).some((key) => {
      const value = object[key]
      if (predicate(value, key, object)) {
        result = key
        return true
      }
    })
    return result
  }`,
  `function findLast(collection, predicate, fromIndex) {
    let iteratee
    const iterable = Object(collection)
    if (!isArrayLike(collection)) {
      collection = Object.keys(collection)
      iteratee = predicate
      predicate = (key) => iteratee(iterable[key], key, iterable)
    }
    const index = findLastIndex(collection, predicate, fromIndex)
    return index > -1 ? iterable[iteratee ? collection[index] : index] : undefined
  }`,
  `function findLastIndex(array, predicate, fromIndex) {
    const length = array == null ? 0 : array.length
    if (!length) {
      return -1
    }
    let index = length - 1
    if (fromIndex !== undefined) {
      index = toInteger(fromIndex)
      index = fromIndex < 0
        ? Math.max(length + index, 0)
        : Math.min(index, length - 1)
    }
    return baseFindIndex(array, predicate, index, true)
  }`,
  `function findLastKey(object, predicate) {
    return baseFindKey(object, predicate, baseForOwnRight)
  }`,
  `function flatMap(collection, iteratee) {
    return baseFlatten(map(collection, iteratee), 1)
  }`,
  `function flattenDeep(array) {
    const length = array == null ? 0 : array.length
    return length ? baseFlatten(array, INFINITY) : []
  }`,
  `function flattenDepth(array, depth) {
    const length = array == null ? 0 : array.length
    if (!length) {
      return []
    }
    depth = depth === undefined ? 1 : +depth
    return baseFlatten(array, depth)
  }`,
  `function flip(func) {
    if (typeof func !== 'function') {
      throw new TypeError('Expected a function')
    }
    return function(...args) {
      return func.apply(this, args.reverse())
    }
  }`,
  `const floor = createRound('floor')`,
  `function flow(...funcs) {
    const length = funcs.length
    let index = length
    while (index--) {
      if (typeof funcs[index] !== 'function') {
        throw new TypeError('Expected a function')
      }
    }
    return function(...args) {
      let index = 0
      let result = length ? funcs[index].apply(this, args) : args[0]
      while (++index < length) {
        result = funcs[index].call(this, result)
      }
      return result
    }
  }`,
  `function flowRight(...funcs) {
    return flow(...funcs.reverse())
  }`,
  `function forEach(collection, iteratee) {
    const func = Array.isArray(collection) ? arrayEach : baseEach
    return func(collection, iteratee)
  }`,
  `function forEachRight(collection, iteratee) {
    const func = Array.isArray(collection) ? arrayEachRight : baseEachRight
    return func(collection, iteratee)
  }`,
  `function forOwn(object, iteratee) {
    object = Object(object)
    Object.keys(object).forEach((key) => iteratee(object[key], key, object))
  }`,
  `function forOwnRight(object, iteratee) {
    if (object == null) {
      return
    }
    const props = Object.keys(object)
    let length = props.length
    while (length--) {
      iteratee(object[props[length]], iteratee, object)
    }
  }`,
  `function fromEntries(pairs) {
    const result = {}
    if (pairs == null) {
      return result
    }
    for (const pair of pairs) {
      result[pair[0]] = pair[1]
    }
    return result
  }`,
  `function functions(object) {
    if (object == null) {
      return []
    }
    return Object.keys(object).filter((key) => typeof object[key] === 'function')
  }`,
  `function get(object, path, defaultValue) {
    const result = object == null ? undefined : baseGet(object, path)
    return result === undefined ? defaultValue : result
  }`,
];

for (let code1 of codes) {
    generate_codes(code1);
}
