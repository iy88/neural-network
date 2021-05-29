import softmax from "./softmax";

export default function diffSoftmax(v: numberArray): numberArray {
  let s = softmax(v);
  let res: numberArray[] = [];
  for (let i = 0; i < v.length; i++) {
    res[i] = [];
    for (let j = 0; j < v.length; j++) {
      if (i === j) {
        res[i][j] = s[i] * (1 - s[j]);
      } else {
        res[i][j] = -s[i] * s[j]
      }
    }
   }
  let d:numberArray = [];
  for(let i = 0; i < v.length; i++){
    // let pd = 0;
    // for(let j = 0; j < res.length; j++){
    //   pd += res[j][i];
    // }
    // d.push(pd);
    d.push(res[i][i])
  }
  return d
  // return v.map(() => 1);
}
