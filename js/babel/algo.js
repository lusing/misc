const coder = require('./gencode');

const codes = [
    `class LinkedListNode {
        constructor(value, next = null) {
          this.value = value;
          this.next = next;
        }
      
        toString(callback) {
          return callback ? callback(this.value) : \`${this.value}\`;
        }
      }`,
]

for (let code1 of codes) {
    coder.generate_codes(code1);
}
