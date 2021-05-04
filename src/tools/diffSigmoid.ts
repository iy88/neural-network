function diffSigmoid(x:number):number{
  function sigmoid(x: number): number {
    return 1/(1+Math.exp(-x));
  }
  return sigmoid(1-sigmoid(x))
}

export default diffSigmoid;