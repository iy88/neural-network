function sigmoid(v: numberArray): numberArray {
  return v.map(z => 1 / (1 + Math.exp(-z)));
}
export default sigmoid;