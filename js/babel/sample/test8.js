"use strict";

var f1 = function f1() {
  var x = 1;
  var y = 2;
  var _ref = [y, x];
  x = _ref[0];
  y = _ref[1];
};

f1();
"use strict";

let f2 = () => {
  let x = 1;
  let y = 2;
  [x, y] = [y, x];
};

f2();
