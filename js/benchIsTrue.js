const isTrue1 = (val) => {
    return val === 'true' || val === true;
};
const isTrue2 = (value) => {
    return String(value) === 'true';
};

function test1() {
    console.time('isTrue1');
    for (let i = 0; i < 100_0000; i++) {
        isTrue1(true);
    }
    console.timeEnd('isTrue1');
}

function test2() {
    console.time('isTrue2');
    for (let i = 0; i < 100_0000; i++) {
        isTrue2('true');
    }
    console.timeEnd('isTrue2');
}

test1(); //about 2ms
test2(); // about 6ms
