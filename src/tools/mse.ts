function mse(yTrue: number, yPred: number): number {
  return (1 / 2) * Math.pow(yTrue - yPred, 2);
}

export default mse;