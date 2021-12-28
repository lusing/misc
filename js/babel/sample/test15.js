"use strict";

function _typeof(obj) { "@babel/helpers - typeof"; return _typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) { return typeof obj; } : function (obj) { return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }, _typeof(obj); }

var f1 = function f1() {
  var s1 = Symbol();
  return _typeof(s1);
};

f1();
"use strict";

let f2 = () => {
  let s1 = Symbol();
  return typeof s1;
};

f2();
