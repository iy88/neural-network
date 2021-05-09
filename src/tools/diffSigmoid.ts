import sigmoid from "./sigmoid";
function diffSigmoid(x:number):number{
  return sigmoid(1-sigmoid(x))
}

export default diffSigmoid;