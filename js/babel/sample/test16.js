require("@babel/polyfill");
"use strict";

function _defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }

var f1 = function f1() {
  var obj1 = _defineProperty({}, Symbol.iterator, /*#__PURE__*/regeneratorRuntime.mark(function _callee() {
    return regeneratorRuntime.wrap(function _callee$(_context) {
      while (1) {
        switch (_context.prev = _context.next) {
          case 0:
            _context.next = 2;
            return 1;

          case 2:
            _context.next = 4;
            return 2;

          case 4:
            _context.next = 6;
            return 3;

          case 6:
          case "end":
            return _context.stop();
        }
      }
    }, _callee);
  }));
};

f1();

