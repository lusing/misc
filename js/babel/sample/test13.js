"use strict";

var f1 = function f1(f2) {
  try {
    f2();
  } catch (_unused) {
    console.error("Error");
  }
};

f1(console.log);
"use strict";

let f3 = f2 => {
  try {
    f2();
  } catch {
    console.error("Error");
  }
};

f3(console.log);
