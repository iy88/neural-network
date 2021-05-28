export default function diffReLU(v: numberArray): numberArray {
  return v.map(z => z > 0 ? 1 : 0);
}