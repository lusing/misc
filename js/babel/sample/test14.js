"use strict";

var f1 = function f1() {
  var a1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  var a2 = [].concat(a1);
};

f1();
"use strict";

let f2 = () => {
  const a1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  let a2 = [...a1];
};

f2();