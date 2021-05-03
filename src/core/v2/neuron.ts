import vector from "../../tools/vector";

class Neuron {
  private weigths: numberArray = [];
  private bias: number = 0;

  constructor(shape: number, weights?: numberArray) {
    if (weights && weights.length !== 0) {
      this.weigths = weights!.slice(0, -1);
      this.bias = weights!.slice(-1)[0];
    } else {
      for (let i = 0; i < shape; i++) {
        this.weigths.push(1);
      }
    }
  }

  feedforward(inputs: numberArray) {
    if (inputs.length === this.weigths.length) {
      return vector.dot(this.weigths, inputs) + this.bias
    } else {
      throw TypeError("inputs shape was not matched");
    }
  }

  backward(inputs: numberArray, derivative: number) {
    let lastWeights = this.weigths.slice(0);
    for (let i = 0; i < this.weigths.length; i++) {
      this.weigths[i] -= derivative * inputs[i];
    }
    this.bias -= derivative;
    return lastWeights;
  }

  snapshot() {
    return { weigths: this.weigths, bias: this.bias };
  }
}


export default Neuron;