const isTrue1 = (val) => { return val === 'true' || val === true; };
const isTrue2 = (value) => { return String(value) === 'true'; };
const isNull = (value) => { return value === null; };
const isUndefined = (value) => { return typeof value === 'undefined'; }
const isNullish1 = (value) => { return isUndefined(value) || isNull(value); };
const isNullish2 = (value) => { return value === null || typeof value === 'undefined'; };
const isNullish3 = (value) => { return value === null || value === undefined; };
const isNullish4 = (value) => {return !(value ?? true)};
const isString = (value) => { return typeof value === 'string'; };
