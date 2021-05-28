function mse(yTrue: number, yPred: number): number {
  return (1 / 2) * Math.pow(yPred - yTrue, 2);
}

export default mse;