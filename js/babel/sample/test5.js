"use strict";

function _readOnlyError(name) { throw new TypeError("\"" + name + "\" is read-only"); }

var f1 = function f1() {
  var a = 0;
  2, _readOnlyError("a");
};
//f1();
"use strict";

let f2 = () => {
  const a = 0;
  a = 2;
};
f2();
