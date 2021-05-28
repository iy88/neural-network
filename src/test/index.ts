import { writeFileSync } from "fs";
// import { readFileSync } from "fs";
import Network from "../core/v2/network";
// import hardmax from "../tools/activationFunctions/hardmax";
let n = new Network('crossEntropy');
n.addLayer({ shape: 2, quant: 10, activationFunction: 'relu' });
n.addLayer({ quant: 1, activationFunction: 'sigmoid' });
let trainingData: { input: numberArray; output: numberArray; }[] = [
  { input: [1, 1], output: [0] },
  { input: [1, 0], output: [1] },
  { input: [0, 1], output: [1] },
  { input: [0, 0], output: [0] }
  // { input: [0], output: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
  // { input: [1], output: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] },
  // { input: [2], output: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] },
  // { input: [3], output: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] },
  // { input: [4], output: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] },
  // { input: [5], output: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] },
  // { input: [6], output: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] },
  // { input: [7], output: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] },
  // { input: [8], output: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] },
  // { input: [9], output: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] },
];
// let testingData: { input: numberArray, output: numberArray }[] = [
// { input: [1, 1], output: [0, 1] },
// { input: [1, 0], output: [1, 0] },
// { input: [0, 1], output: [1, 0] },
// { input: [0, 0], output: [0, 1] }
// ]
// let loss = n.fit(trainingData, testingData, .70, Infinity, 10000);
let loss = n.train(trainingData, 0.1, 100000);
writeFileSync('./loss', loss.join('\n'), { flag: 'w+' });
// n.load(JSON.parse(readFileSync('./model.json').toString()));
for (let i of trainingData) {
  let inp = i.input;
  let nop = n.feedforward(inp);
  console.log(inp, '\n', nop, '\n------------');
}
// writeFileSync('./model.json', JSON.stringify(n.snapshot()));