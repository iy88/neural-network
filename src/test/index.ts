import { writeFileSync } from "fs";
import { readFileSync } from "fs";
import Network from "../core/v2/network";
// import diffReLU from "../tools/diffReLu";
// import LeakyReLU from "../tools/LeakyReLU";
// import diffReLU from "../tools/diffReLu";
// import diffSigmoid from "../tools/diffSigmoid";
// import ReLU from "../tools/ReLU";
// import sigmoid from "../tools/sigmoid";
let n = new Network();
n.addLayer(2, 10);
n.addLayer(10, 1);
let trainingData: { input: numberArray; output: numberArray; }[] = [
  { input: [1,2], output: [(1+2)/2] },
  { input: [3,4], output: [(3+4)/2] },
  { input: [5,6], output: [(5+6)/2] },
]
let testingData:{input:numberArray,output:numberArray}[] = [
  { input: [7,8], output: [(7+8)/2] },
  { input: [9,10], output: [(9+10)/2] },
  { input: [11,12], output: [(11+12)/2] },
]
console.log('-----raw-----');
console.time()
console.log(n.fit(trainingData, testingData, 1e-3, Math.max(trainingData.length,testingData.length), 10000).length);
writeFileSync('./model.json', JSON.stringify(n.snapshot()));
console.timeEnd();
console.log(n.feedforward([1,2])[0],(1+2)/2);
console.log(n.feedforward([2,3])[0],(2+3)/2);
console.log('-----copy-----');
let file = readFileSync('./model.json').toString()
n.load(JSON.parse(file));
console.log(n.feedforward([1,2])[0],(1+2)/2);
console.log(n.feedforward([2,3])[0],(2+3)/2);