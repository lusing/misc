"use strict";

function greet(input) {
  return input ?? "Hello world";
}
greet();
let f1 = () => {let [a, b, c] = [1, 2, 3];};
f1();

var f2 = function f2() {
  var a = 0;
};

f2();

let f3 = () => {
  let a = 0;
};
f3();
