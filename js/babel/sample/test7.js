"use strict";

var f1 = function f1() {
  var _ = void 0,
      a = _ === void 0 ? 1 : _;
};

f1();
"use strict";

let f2 = () => {
  let [a = 1] = [void 0];
};

f2();
