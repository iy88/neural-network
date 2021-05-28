export default function hardmax(x: numberArray): numberArray {
  let max = Math.max(...x);
  return x.map(x => x === max ? 1 : 0);
}