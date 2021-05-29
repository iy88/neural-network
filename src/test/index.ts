import { writeFileSync } from "fs";
import { readFileSync } from "fs";
import Network from "../core/v2/network";
// import hardmax from "../tools/activationFunctions/hardmax";
let n = new Network('mse');
n.addLayer({ shape: 1, quant: 10, activationFunction: 'relu' });
n.addLayer({ quant: 10, activationFunction: 'softmax' });
let trainingData: { input: numberArray; output: numberArray; }[] = [
  // { input: [1, 1], output: [0,1] },
  // { input: [1, 0], output: [1,0] },
  // { input: [0, 1], output: [1,0] },
  // { input: [0, 0], output: [0,1] }
  // { input: [1, 1], output: [2] },
  // { input: [1, 2], output: [3] },
  // { input: [2, 3], output: [5] },
  // { input: [3, 5], output: [8] }
  { input: [0], output: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
  { input: [1], output: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] },
  { input: [2], output: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] },
  { input: [3], output: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] },
  { input: [4], output: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] },
  { input: [5], output: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] },
  { input: [6], output: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] },
  { input: [7], output: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] },
  { input: [8], output: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] },
  { input: [9], output: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] },
];
// let testingData: { input: numberArray, output: numberArray }[] = [
//   { input: [0], output: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
//   { input: [1], output: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] },
//   { input: [2], output: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] },
//   { input: [3], output: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] },
//   { input: [4], output: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] },
//   { input: [5], output: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] },
//   { input: [6], output: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] },
//   { input: [7], output: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] },
//   { input: [8], output: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] },
//   { input: [9], output: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] },
  // { input: [1, 2], output: [3] },
  // { input: [2, 2], output: [4] },
  // { input: [2, 5], output: [7] },
  // { input: [3, 6], output: [9] }
// ]
// let loss = n.fit(trainingData, testingData, .70, Infinity, 10000);
n.load(JSON.parse(readFileSync('./model.json').toString()));
// let loss = n.train(trainingData, 1, 10);
n.train(trainingData, 1000, 10000);
// writeFileSync('./loss', loss.join('\n'), { flag: 'w+' });
for (let i of trainingData) {
  let inp = i.input;
  let nop = n.feedforward(inp);
  console.log(inp, '\n', nop, '\n------------');
}
writeFileSync('./model.json', JSON.stringify(n.snapshot()));