import vector from "../../tools/vector";

class Neuron {
  private weights: numberArray = [];
  private bias: number = 0;

  constructor(shape: number, weights?: numberArray) {
    if (weights && weights.length !== 0) {
      this.weights = weights!.slice(0, -1);
      this.bias = weights!.slice(-1)[0];
    } else {
      for (let i = 0; i < shape; i++) {
        this.weights.push(Math.random());
      }
    }
  }

  feedforward(inputs: numberArray) {
    if (inputs.length === this.weights.length) {
      return vector.dot(this.weights, inputs) + this.bias
    } else {
      throw TypeError("inputs shape was not matched");
    }
  }

  backward(inputs: numberArray, derivative: number) {
    let lastWeights = this.weights.slice(0);
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] -= derivative * inputs[i];
    }
    this.bias -= derivative;
    return lastWeights;
  }

  snapshot() {
    return { weigths: this.weights, bias: this.bias };
  }
}


export default Neuron;
module.exports = Neuron;