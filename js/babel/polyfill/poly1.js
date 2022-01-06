let polyfill1 = () => {
    Array.from(new Set([1, 2, 3, 2, 1]));
    [1, [2, 3], [4, [5]]].flat(2);
    Promise.resolve(32).then(x => console.log(x));
}
polyfill1();
