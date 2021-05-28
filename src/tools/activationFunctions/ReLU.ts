export default function ReLU(v: numberArray): numberArray {
  // return Math.max(0, z)
  return v.map(z => Math.max(0, z));
}