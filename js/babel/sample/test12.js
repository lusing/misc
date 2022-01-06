var f1 = function f1() {
    var sum = 0;

    for (var _len = arguments.length, values = new Array(_len), _key = 0; _key < _len; _key++) {
        values[_key] = arguments[_key];
    }

    for (var _i = 0, _values = values; _i < _values.length; _i++) {
        var v = _values[_i];
        sum += v;
    }

    return sum;
};

f1(1, 4, 9);

let f2 = (...values) => {
    let sum = 0;
    for (let v of values) {
        sum += v;
    }
    return sum;
};
f2(1, 4, 9);