"use strict";

let f2 = () => {
  let obj1 = {
    *[Symbol.iterator]() {
      yield 1;
      yield 2;
      yield 3;
    }
  };
  [...obj1]
};

f2();
