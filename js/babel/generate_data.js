const sourceMap = require('source-map');
const fs = require('fs');

const rawSourceMap = {
    version: 3,
    file: "min.js",
    names: ["bar", "baz", "n"],
    sources: ["one.js", "two.js"],
    sourceRoot: "http://example.com/www/js/",
    mappings: "CAAC,IAAI,IAAM,SAAUA,GAClB,OAAOC,IAAID;CCDb,IAAI,IAAM,SAAUE,GAClB,OAAOA"
};

const sourceMap2 = { "version": 3, "sources": ["../test1.js"], "names": ["f1", "Code", "source", "code1"], "mappings": ";;;;;;;;AAAA,IAAIA,EAAE,GAAG,SAALA,EAAK,GAAM;AAAA,MACLC,IADK,6BAEP,cAAYC,MAAZ,EAAoB;AAAA;;AAChB,SAAKA,MAAL,GAAcA,MAAd;AACH,GAJM;;AAMXC,EAAAA,KAAK,GAAG,IAAIF,IAAJ,CAAS,UAAT,CAAR;AACH,CAPD;;AAQAD,EAAE", "sourcesContent": ["let f1 = () => {\n    class Code {\n        constructor(source) {\n            this.source = source;\n        }\n    }\n    code1 = new Code(\"test1.js\");\n};\nf1();\n"], "file": "test1.js" }

function get_lines(all_code, line_number) {
    const lines = all_code.split('\n');
    return lines[line_number];
}

const whatever = sourceMap.SourceMapConsumer.with(sourceMap2, null, consumer => {
    console.log(consumer.file);
    let filename = consumer.file;
    let all_code = fs.readFileSync(filename,{encoding:'utf8'});
    const lines = all_code.split('\n');
    let result = new Set();
    // [ 'http://example.com/www/js/one.js',
    //   'http://example.com/www/js/two.js' ]

    // console.log(
    //     consumer.originalPositionFor({
    //         line: 2,
    //         column: 28
    //     })
    // );
    // { source: 'http://example.com/www/js/two.js',
    //   line: 2,
    //   column: 10,
    //   name: 'n' }

    // console.log(
    //     consumer.generatedPositionFor({
    //         source: "http://example.com/www/js/two.js",
    //         line: 2,
    //         column: 10
    //     })
    // );
    // { line: 2, column: 28 }

    consumer.eachMapping(function (m) {
        //console.log(m);
        let line = lines[m.generatedLine-1];
        console.log(line);
        console.log(line[m.generatedColumn]);
        //console.log
        result.add({to:m.generatedLine, from: m.originalLine});
    });

    console.log(result);

    return result;
});
