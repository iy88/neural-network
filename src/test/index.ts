import Network from "../core/v2/network";
import diffSigmoid from "../tools/diffSigmoid";
import sigmoid from "../tools/sigmoid";
import path from "path";
import fs from "fs";
let n = new Network();
n.addLayer(1, 3);
n.addLayer(3,2)
n.addLayer(2, 1,sigmoid,diffSigmoid);
let trainingData: { input: numberArray, output: numberArray }[] = [];
let testingData: { input: numberArray, output: numberArray }[] = []
for (let i = 40; i <= 70; i++) {
  trainingData.push({ input: [i], output: [i >= 60 ? 1 : 0] });
}
for (let i = 71; i <= 80; i++) {
  testingData.push({ input: [i], output: [i >= 60 ? 1 : 0] });
}
// let loss = n.train(trainingData, 1e-3, 10000, true, 10, testingData);
n.fit(trainingData,testingData,1e-5,0);
// console.log(loss.slice(loss.length - 10))
// console.log(n.snapshot())
let model:anyObject[] = n.snapshot();
// let n2 = new Network();
// n2.load(model);
// fs.writeFileSync(path.resolve(path.join(__dirname,'./model')),JSON.stringify(model));
let stream = fs.createWriteStream(path.resolve(path.join(__dirname,'./model')),{flags:'w+'});
stream.write(JSON.stringify(model));
stream.end();
console.log(n.feedforward([100]));
