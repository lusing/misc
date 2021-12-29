const coder = require('./gencode');

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
    `function groupBy(collection, iteratee) {
    return reduce(collection, (result, value, key) => {
      key = iteratee(value)
      if (hasOwnProperty.call(result, key)) {
        result[key].push(value)
      } else {
        baseAssignValue(result, key, [value])
      }
      return result
    }, {})
  }`,
    `function gt(value, other) {
    if (!(typeof value === 'string' && typeof other === 'string')) {
      value = +value
      other = +other
    }
    return value > other
  }`,
    `function gte(value, other) {
    if (!(typeof value === 'string' && typeof other === 'string')) {
      value = +value
      other = +other
    }
    return value >= other
  }`,
    `function has(object, key) {
    return object != null && hasOwnProperty.call(object, key)
  }`,
    `function hasIn(object, key) {
    return object != null && key in Object(object)
  }`,
    `function hasPath(object, path) {
    path = castPath(path, object)
  
    let index = -1
    let { length } = path
    let result = false
    let key
  
    while (++index < length) {
      key = toKey(path[index])
      if (!(result = object != null && hasOwnProperty.call(object, key))) {
        break
      }
      object = object[key]
    }
    if (result || ++index != length) {
      return result
    }
    length = object == null ? 0 : object.length
    return !!length && isLength(length) && isIndex(key, length) &&
      (Array.isArray(object) || isArguments(object))
  }`,
    `function hasPathIn(object, path) {
    path = castPath(path, object)
  
    let index = -1
    let { length } = path
    let result = false
    let key
  
    while (++index < length) {
      key = toKey(path[index])
      if (!(result = object != null && key in Object(object))) {
        break
      }
      object = object[key]
    }
    if (result || ++index != length) {
      return result
    }
    length = object == null ? 0 : object.length
    return !!length && isLength(length) && isIndex(key, length) &&
      (Array.isArray(object) || isArguments(object))
  }`,
    `function head(array) {
    return (array != null && array.length)
      ? array[0]
      : undefined
  }`,
    `function inRange(number, start, end) {
    if (end === undefined) {
      end = start
      start = 0
    }
    return baseInRange(+number, +start, +end)
  }`,
    `function indexOf(array, value, fromIndex) {
    const length = array == null ? 0 : array.length
    if (!length) {
      return -1
    }
    let index = fromIndex == null ? 0 : toInteger(fromIndex)
    if (index < 0) {
      index = Math.max(length + index, 0)
    }
    return baseIndexOf(array, value, index)
  }`,
    `function initial(array) {
    const length = array == null ? 0 : array.length
    return length ? slice(array, 0, -1) : []
  }`,
    `function intersection(...arrays) {
    const mapped = map(arrays, castArrayLikeObject)
    return (mapped.length && mapped[0] === arrays[0])
      ? baseIntersection(mapped)
      : []
  }`,
    `function intersectionBy(...arrays) {
    let iteratee = last(arrays)
    const mapped = map(arrays, castArrayLikeObject)
  
    if (iteratee === last(mapped)) {
      iteratee = undefined
    } else {
      mapped.pop()
    }
    return (mapped.length && mapped[0] === arrays[0])
      ? baseIntersection(mapped, iteratee)
      : []
  }`,
    `function intersectionWith(...arrays) {
    let comparator = last(arrays)
    const mapped = map(arrays, castArrayLikeObject)
  
    comparator = typeof comparator === 'function' ? comparator : undefined
    if (comparator) {
      mapped.pop()
    }
    return (mapped.length && mapped[0] === arrays[0])
      ? baseIntersection(mapped, undefined, comparator)
      : []
  }`,
    `function invert(object) {
    const result = {}
    Object.keys(object).forEach((key) => {
      let value = object[key]
      if (value != null && typeof value.toString !== 'function') {
        value = toString.call(value)
      }
      result[value] = key
    })
    return result
  }`,
    `function invertBy(object, iteratee) {
    const result = {}
    Object.keys(object).forEach((key) => {
      const value = iteratee(object[key])
      if (hasOwnProperty.call(result, value)) {
        result[value].push(key)
      } else {
        result[value] = [key]
      }
    })
    return result
  }`,
    `function invoke(object, path, args) {
    path = castPath(path, object)
    object = parent(object, path)
    const func = object == null ? object : object[toKey(last(path))]
    return func == null ? undefined : func.apply(object, args)
  }`,
    `function invokeMap(collection, path, args) {
    let index = -1
    const isFunc = typeof path === 'function'
    const result = isArrayLike(collection) ? new Array(collection.length) : []
  
    baseEach(collection, (value) => {
      result[++index] = isFunc ? path.apply(value, args) : invoke(value, path, args)
    })
    return result
  }`,
    `function isArguments(value) {
    return isObjectLike(value) && getTag(value) == '[object Arguments]'
  }`,
    `const isArrayBuffer = nodeIsArrayBuffer
  ? (value) => nodeIsArrayBuffer(value)
  : (value) => isObjectLike(value) && getTag(value) == '[object ArrayBuffer]'`,
    `const isArrayBuffer = nodeIsArrayBuffer
  ? (value) => nodeIsArrayBuffer(value)
  : (value) => isObjectLike(value) && getTag(value) == '[object ArrayBuffer]'`,
    `function isArrayLike(value) {
    return value != null && typeof value !== 'function' && isLength(value.length)
  }`,
    `function isArrayLikeObject(value) {
    return isObjectLike(value) && isArrayLike(value)
  }`,
    `function isBoolean(value) {
    return value === true || value === false ||
      (isObjectLike(value) && getTag(value) == '[object Boolean]')
  }`,
    `const isBuffer = nativeIsBuffer || (() => false)`,
    `const nativeIsBuffer = Buffer ? Buffer.isBuffer : undefined`,
    `const isDate = nodeIsDate
  ? (value) => nodeIsDate(value)
  : (value) => isObjectLike(value) && getTag(value) == '[object Date]'`,
    `function isElement(value) {
    return isObjectLike(value) && value.nodeType === 1 && !isPlainObject(value)
  }`,
    `function isEmpty(value) {
    if (value == null) {
      return true
    }
    if (isArrayLike(value) &&
        (Array.isArray(value) || typeof value === 'string' || typeof value.splice === 'function' ||
          isBuffer(value) || isTypedArray(value) || isArguments(value))) {
      return !value.length
    }
    const tag = getTag(value)
    if (tag == '[object Map]' || tag == '[object Set]') {
      return !value.size
    }
    if (isPrototype(value)) {
      return !Object.keys(value).length
    }
    for (const key in value) {
      if (hasOwnProperty.call(value, key)) {
        return false
      }
    }
    return true
  }`,
    `function isEqualWith(value, other, customizer) {
    customizer = typeof customizer === 'function' ? customizer : undefined
    const result = customizer ? customizer(value, other) : undefined
    return result === undefined ? baseIsEqual(value, other, undefined, customizer) : !!result
  }`,
    `function isError(value) {
    if (!isObjectLike(value)) {
      return false
    }
    const tag = getTag(value)
    return tag == '[object Error]' || tag == '[object DOMException]' ||
      (typeof value.message === 'string' && typeof value.name === 'string' && !isPlainObject(value))
  }`,
    `function isFunction(value) {
    return typeof value === 'function'
  }`,
    `function isLength(value) {
    return typeof value === 'number' &&
      value > -1 && value % 1 == 0 && value <= MAX_SAFE_INTEGER
  }`,
    `const isMap = nodeIsMap
  ? (value) => nodeIsMap(value)
  : (value) => isObjectLike(value) && getTag(value) == '[object Map]'`,
    `function isMatch(object, source) {
    return object === source || baseIsMatch(object, source, getMatchData(source))
  }`,
    `function isMatchWith(object, source, customizer) {
    customizer = typeof customizer === 'function' ? customizer : undefined
    return baseIsMatch(object, source, getMatchData(source), customizer)
  }`,
    `
  function isNative(value) {
    return isObject(value) && reIsNative.test(value)
  }
  `,
    `function isNil(value) {
    return value == null
  }`,
    `function isNull(value) {
    return value === null
  }`,
    `function isNumber(value) {
    return typeof value === 'number' ||
      (isObjectLike(value) && getTag(value) == '[object Number]')
  }`,
    `function isObject(value) {
    const type = typeof value
    return value != null && (type === 'object' || type === 'function')
  }`,
    `function isObjectLike(value) {
    return typeof value === 'object' && value !== null
  }`,
    `function isPlainObject(value) {
    if (!isObjectLike(value) || getTag(value) != '[object Object]') {
      return false
    }
    if (Object.getPrototypeOf(value) === null) {
      return true
    }
    let proto = value
    while (Object.getPrototypeOf(proto) !== null) {
      proto = Object.getPrototypeOf(proto)
    }
    return Object.getPrototypeOf(value) === proto
  }`,
    `const isRegExp = nodeIsRegExp
  ? (value) => nodeIsRegExp(value)
  : (value) => isObjectLike(value) && getTag(value) == '[object RegExp]'`,
    `const isSet = nodeIsSet
  ? (value) => nodeIsSet(value)
  : (value) => isObjectLike(value) && getTag(value) == '[object Set]'`,
    `function isString(value) {
    const type = typeof value
    return type === 'string' || (type === 'object' && value != null && !Array.isArray(value) && getTag(value) == '[object String]')
  }`,
    `function isSymbol(value) {
    const type = typeof value
    return type == 'symbol' || (type === 'object' && value != null && getTag(value) == '[object Symbol]')
  }`,
    `const isTypedArray = nodeIsTypedArray
  ? (value) => nodeIsTypedArray(value)
  : (value) => isObjectLike(value) && reTypedTag.test(getTag(value))`,
    `function isUndefined(value) {
    return value === undefined
  }`,
    `function isWeakMap(value) {
    return isObjectLike(value) && getTag(value) == '[object WeakMap]'
  }`,
    `function isWeakSet(value) {
    return isObjectLike(value) && getTag(value) == '[object WeakSet]'
  }`,
    `const kebabCase = (string) => (
    words(toString(string).replace(/['\u2019]/g, '')).reduce((result, word, index) => (
      result + (index ? '-' : '') + word.toLowerCase()
    ), '')
  )`,
    `function keyBy(collection, iteratee) {
    return reduce(collection, (result, value, key) => (
      baseAssignValue(result, iteratee(value), value), result
    ), {})
  }`,
    `function keys(object) {
    return isArrayLike(object)
      ? arrayLikeKeys(object)
      : Object.keys(Object(object))
  }`,
    `function keysIn(object) {
    const result = []
    for (const key in object) {
      result.push(key)
    }
    return result
  }`,
    `function last(array) {
    const length = array == null ? 0 : array.length
    return length ? array[length - 1] : undefined
  }`,
    `function lastIndexOf(array, value, fromIndex) {
    const length = array == null ? 0 : array.length
    if (!length) {
      return -1
    }
    let index = length
    if (fromIndex !== undefined) {
      index = toInteger(fromIndex)
      index = index < 0 ? Math.max(length + index, 0) : Math.min(index, length - 1)
    }
    return value === value
      ? strictLastIndexOf(array, value, index)
      : baseFindIndex(array, baseIsNaN, index, true)
  }`,
    `const lowerCase = (string) => (
    words(toString(string).replace(reQuotes, '')).reduce((result, word, index) => (
      result + (index ? ' ' : '') + word.toLowerCase()
    ), '')
  )`,
    `const lowerFirst = createCaseFirst('toLowerCase')`,
    `function lt(value, other) {
    if (!(typeof value === 'string' && typeof other === 'string')) {
      value = +value
      other = +other
    }
    return value < other
  }`,
    `function lte(value, other) {
    if (!(typeof value === 'string' && typeof other === 'string')) {
      value = +value
      other = +other
    }
    return value <= other
  }
  `,
    `function map(array, iteratee) {
    let index = -1
    const length = array == null ? 0 : array.length
    const result = new Array(length)
  
    while (++index < length) {
      result[index] = iteratee(array[index], index, array)
    }
    return result
  }
  `,
    `function mapKey(object, iteratee) {
    object = Object(object)
    const result = {}
  
    Object.keys(object).forEach((key) => {
      const value = object[key]
      result[iteratee(value, key, object)] = value
    })
    return result
  }`,
    `function mapObject(object, iteratee) {
    const props = Object.keys(object)
    const result = new Array(props.length)
  
    props.forEach((key, index) => {
      result[index] = iteratee(object[key], key, object)
    })
    return result
  }`,
    `function mapValue(object, iteratee) {
    object = Object(object)
    const result = {}
  
    Object.keys(object).forEach((key) => {
      result[key] = iteratee(object[key], key, object)
    })
    return result
  }`,
    `import baseAt from './.internal/baseAt.js'
  import baseFlatten from './.internal/baseFlatten.js'
  
  const at = (object, ...paths) => baseAt(object, baseFlatten(paths, 1))
  
  export default at`,
    `function matches(source) {
  return baseMatches(baseClone(source, CLONE_DEEP_FLAG))
}`,
    `function matchesProperty(path, srcValue) {
        return baseMatchesProperty(path, baseClone(srcValue, CLONE_DEEP_FLAG))
      }`,
`function maxBy(array, iteratee) {
    let result
    if (array == null) {
      return result
    }
    let computed
    for (const value of array) {
      const current = iteratee(value)
  
      if (current != null && (computed === undefined
        ? (current === current && !isSymbol(current))
        : (current > computed)
      )) {
        computed = current
        result = value
      }
    }
    return result
  }`,
  `function mean(array) {
    return baseMean(array, (value) => value)
  }`,
  `function meanBy(array, iteratee) {
    const length = array == null ? 0 : array.length
    return length ? (baseSum(array, iteratee) / length) : NAN
  }`,
  `function memoize(func, resolver) {
    if (typeof func !== 'function' || (resolver != null && typeof resolver !== 'function')) {
      throw new TypeError('Expected a function')
    }
    const memoized = function(...args) {
      const key = resolver ? resolver.apply(this, args) : args[0]
      const cache = memoized.cache
  
      if (cache.has(key)) {
        return cache.get(key)
      }
      const result = func.apply(this, args)
      memoized.cache = cache.set(key, result) || cache
      return result
    }
    memoized.cache = new (memoize.Cache || Map)
    return memoized
  }`,
  `const merge = createAssigner((object, source, srcIndex) => {
    baseMerge(object, source, srcIndex)
  })
  `,
  `function method(path, args) {
    return (object) => invoke(object, path, args)
  }`,
  `function nth(array, n) {
    const length = array == null ? 0 : array.length
    if (!length) {
      return
    }
    n += n < 0 ? length : 0
    return isIndex(n, length) ? array[n] : undefined
  }`,
  `function once(func) {
    return before(2, func)
  }`,
];

for (let code1 of codes) {
    coder.generate_codes(code1);
}
