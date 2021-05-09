import { writeFileSync,readFileSync} from "fs";
import Network from "../core/v2/network";
import sum from "../tools/sum";
let n = new Network('crossEntropy');
n.addLayer(2,2,'relu');
n.addLayer(2,2,'relu');
n.addLayer(2,1,'sigmoid');
let trainingData: { input: numberArray; output: numberArray; }[] = [
  {input:[1,0],output:[1]},
  {input:[1,1],output:[0]},
  {input:[0,1],output:[1]},
  {input:[0,0],output:[0]}
  // {input:[1],output:[1]},
  // {input:[2],output:[2]},
];
// for (let i = 0; i < 10; i++) {
//   // for (let j = 0; j < 10; j++) {
//   //   trainingData.push({ input: [i, j], output: [i + j] });
//   // }
//   trainingData.push({input:[i],output:[i % 2 === 0 ? 1 : 0]});
// }
// console.log(trainingData);
console.time();
writeFileSync('./loss',n.fit(trainingData,trainingData,1e-2,1000000,1000000).join('\n'));
// n.train(trainingData,1e-4,1000000);
// writeFileSync('./loss',loss.map(l=>l.join(' ')).join('\n'));
console.timeEnd();
writeFileSync('./model.json', JSON.stringify(n.snapshot()));
n.load(JSON.parse(readFileSync('./model.json').toString()));
let losses: numberArray = [];
for (let i = 0; i < trainingData.length; i++) {
  let pred = n.feedforward(trainingData[i].input)[0];
  let real = trainingData[i].output[0];
  losses.push(Math.pow(pred - real, 2));
}
let mse = sum(losses) / losses.length;
console.log("mse", mse);
for(let i of trainingData){
  console.log(n.feedforward(i.input));
}