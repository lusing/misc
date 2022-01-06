"use strict";

var f1 = function f1() {
  var a1 = 100000000;
  var a2 = 100000000n;
};
f1();
"use strict";

let f2 = () => {
  let a1 = 100_000_000;
  let a2 = 100_000_000n;
};
f2();
