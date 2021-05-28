import sum from "../sum"
export default function softmax(v: numberArray): numberArray {
  let d = Math.max(...v);
  let a = v.map(z => Math.exp(z-d));
  let s = sum(a);
  return a.map(z => z / s);
}