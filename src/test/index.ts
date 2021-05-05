import { writeFileSync } from "fs";
import Network from "../core/v2/network";
import sum from "../tools/sum";
let n = new Network();
n.addLayer(2, 10);
n.addLayer(10, 1);
let trainingData: { input: numberArray; output: numberArray; }[] = [];
for (let i = 0; i < 100; i++) {
  let x = Math.floor(Math.random() * 10);
  let y = Math.floor(Math.random() * 10);
  trainingData.push({ input: [x, y], output: [x + y] })
}
console.log('-----raw-----');
console.time()
n.train(trainingData, 1e-3, 10000);
writeFileSync('./model.json', JSON.stringify(n.snapshot()));
console.timeEnd();
let acts: numberArray = [];
for (let i = 0; i < 10000; i++) {
  let x = Math.floor(Math.random() * 10);
  let y = Math.floor(Math.random() * 10);
  let tr = x + y;
  let act = (1 - ((n.feedforward([x, y])[0] - tr)) / tr);
  isFinite(act) ? acts.push(act) : i--;
}
console.log("accuracy",sum(acts) / acts.length * 100);
console.log(n.feedforward([1, 1])[0]);