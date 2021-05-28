import sigmoid from "./sigmoid";
import vector from "../vector";
function diffSigmoid(v: numberArray): numberArray {
  let pr1 = vector.linearCombinations([[1,v.map(()=>1)],[-1,sigmoid(v)]]);
  return sigmoid(pr1);
}

export default diffSigmoid;