import sum from "./sum";

function dot(x: numberArray | numberArray[], y: numberArray): number {
  if (x.length === y.length && x.length > 0 && y.length > 0 && typeof x[0] === 'number') {
    return sum((x as numberArray).map((x: number, i: number) => (x * (y as numberArray)[i])));
  } else {
    throw TypeError("x, y was not fit");
  }
}

function addition(V:numberArray[]):numberArray {
  let result = V[0];
  for (let i = 1; i < V.length; i++) {
    for (let j = 0; j < V[i].length; j++) {
      result[j] += V[i][j];
    }
  }
  return result;
}

function scalarMultiplication(scalar:number, v:numberArray):numberArray {
  let result = JSON.parse(JSON.stringify(v));
  for (let i = 0; i < v.length; i++) {
    result[i] *= scalar;
  }
  return result;
}

type scalarAndVector = [number,numberArray]

function linearCombinations(scalarAndVectors:scalarAndVector[]):numberArray {
  let result = scalarMultiplication(...scalarAndVectors[0]);
  for (let i = 1; i < scalarAndVectors.length; i++) {
    result = addition([result, scalarMultiplication(...scalarAndVectors[i])]);
  }
  return result;
}

export default {
  dot,
  addition,
  scalarMultiplication,
  linearCombinations
};