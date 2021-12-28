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
  }`
];

for (let code1 of codes) {
    generate_codes(code1);
}
