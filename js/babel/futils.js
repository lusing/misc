const coder = require('./gencode');

const codes = [
    `let isTraversable = x => _.isArray(x) || _.isPlainObject(x)`,
    `let traverse = x => isTraversable(x) && !_.isEmpty(x) && x`,
    `let walk = (next = traverse) => (
        pre,
        post = _.noop,
        parents = [],
        parentIndexes = []
      ) => (tree, index) =>
        pre(tree, index, parents, parentIndexes) ||
        findIndexed(
          walk(next)(pre, post, [tree, ...parents], [index, ...parentIndexes]),
          next(tree, index, parents, parentIndexes) || []
        ) ||
        post(tree, index, parents, parentIndexes)`,
    `let findIndexedAsync = (f, data, remaining = _.toPairs(data)) => {
        if (!remaining.length) return
        let [[key, val], ...rest] = remaining
        return Promise.resolve(f(val, key, data)).then(result =>
          result ? val : rest.length ? findIndexedAsync(f, data, rest) : undefined
        )
      }`,
    `let walkAsync = (next = traverse) => (
        pre,
        post = _.noop,
        parents = [],
        parentIndexes = []
      ) => (tree, index) =>
        Promise.resolve(pre(tree, index, parents, parentIndexes))
          .then(
            preResult =>
              preResult ||
              findIndexedAsync(
                walkAsync(next)(
                  pre,
                  post,
                  [tree, ...parents],
                  [index, ...parentIndexes]
                ),
                next(tree, index, parents, parentIndexes) || []
              )
          )
          .then(stepResult => stepResult || post(tree, index, parents, parentIndexes))`,
    `let transformTree = (next = traverse) =>
    _.curry((f, x) => {
      let result = _.cloneDeep(x)
      walk(next)(f)(result)
      return result
    })`,
    `let reduceTree = (next = traverse) =>
    _.curry((f, result, tree) => {
      walk(next)((...x) => {
        result = f(result, ...x)
      })(tree)
      return result
    })
  `,
    `let writeProperty = (next = traverse) => (node, index, [parent]) => {
    next(parent)[index] = node
  }`,
    `let mapTree = (next = traverse, writeNode = writeProperty(next)) =>
    _.curry(
      (mapper, tree) =>
        transformTree(next)((node, i, parents, ...args) => {
          if (parents.length)
            writeNode(mapper(node, i, parents, ...args), i, parents, ...args)
        })(mapper(tree)) // run mapper on root, and skip root in traversal
    )`,
    `let mapTreeLeaves = (next = traverse, writeNode = writeProperty(next)) =>
    _.curry((mapper, tree) =>
      // this unless wrapping can be done in user land, this is pure convenience
      // mapTree(next, writeNode)(F.unless(next, mapper), tree)
      mapTree(next, writeNode)(node => (next(node) ? node : mapper(node)), tree)
    )`,
];

for (let code1 of codes) {
    coder.generate_codes(code1);
}
