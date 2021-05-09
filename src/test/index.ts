import { writeFileSync,readFileSync} from "fs";
import Network from "../core/v2/network";
import sum from "../tools/sum";
let n = new Network();
n.addLayer(2, 3,'relu');
n.addLayer(3, 2,'relu');
n.addLayer(2, 1,'relu');
n.addLayer(1, 2,'relu');
n.addLayer(2, 3,'relu');
n.addLayer(3, 2,'relu');
n.addLayer(2, 1,'relu');
let trainingData: { input: numberArray; output: numberArray; }[] = [];
for (let i = 0; i < 10; i++) {
  for (let j = 0; j < 10; j++) {
    trainingData.push({ input: [i, j], output: [i + j] });
  }
}
console.time();
n.train(trainingData, 1e-3, 20000);
console.timeEnd();
writeFileSync('./model.json', JSON.stringify(n.snapshot()));
console.log(n.feedforward([100, 100])[0]);
n.load(JSON.parse(readFileSync('./model.json').toString()));
console.log(n.feedforward([100, 100])[0]);
let losses: numberArray = [];
for (let i = 0; i < trainingData.length; i++) {
  let pred = n.feedforward(trainingData[i].input)[0];
  let real = trainingData[i].output[0];
  losses.push(Math.pow(pred - real, 2));
}
let mse = sum(losses) / losses.length;
console.log("mse", mse);
// console.log(losses);
console.log(n.feedforward([100, 100])[0]);