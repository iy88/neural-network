import { writeFileSync } from "fs";
import Network from "../core/v2/network";
// import sum from "../tools/sum";
let n = new Network('crossEntropy');
// n.addLayer(1, 2, 'relu');
// n.addLayer(2, 2, 'relu');
n.addLayer(1, 100,'relu');
n.addLayer(100, 100,'relu');
n.addLayer(100, 100,'relu');
n.addLayer(100, 1,'relu');
let trainingData: { input: numberArray; output: numberArray; }[] = [
  { input: [1], output: [1] },
  { input: [2], output: [2] },
  { input: [3], output: [3] },
  { input: [4], output: [4] }
];
let testingData:{input:numberArray,output:numberArray}[] = [
  { input: [5], output: [5] },
  { input: [6], output: [6] },
  { input: [7], output: [7] },
  { input: [8], output: [8] }
]
// for (let i = 0; i < 10; i++) {
//   // for (let j = 0; j < 10; j++) {
//   //   trainingData.push({ input: [i, j], output: [i + j] });
//   // }
//   trainingData.push({input:[i],output:[i % 2 === 0 ? 1 : 0]});
// }
// console.log(trainingData);
// console.time();
// n.train(trainingData,1e-3,1000);
writeFileSync('./loss',n.fit(trainingData,testingData,1e-3,Infinity,1).join('\n'));
// n.fit(trainingData,trainingData,1e-3,Infinity,10000);
// writeFileSync('./loss',loss.map(l=>l.join(' ')).join('\n'));
// console.timeEnd();
writeFileSync('./model.json', JSON.stringify(n.snapshot()));
// n.load(JSON.parse(readFileSync('./model.json').toString()));
// let losses: numberArray = [];
// for (let i = 0; i < trainingData.length; i++) {
//   let pred = n.feedforward(trainingData[i].input)[0];
//   let real = trainingData[i].output[0];
//   losses.push(Math.pow(pred - real, 2));
// }
// let mse = sum(losses) / losses.length;
// console.log("mse", mse);
for (let i of trainingData) {
  console.log(n.feedforward(i.input), i.output);
}