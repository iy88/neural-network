import sum from "../../tools/sum";
import Neuron from "./neuron";

class Layer {
  private neurons: Neuron[] = [];
  private activationFunction: Function | void = undefined
  private derivativeOfActivationFunction: Function | void = undefined;
  private lastOut: numberArray = [];
  private shape: number = 0;

  constructor(shape: number, quant: number, activationFunction?: Function, derivativeOfActivationFunction?: Function, neurons?: Neuron[]) {
    this.activationFunction = activationFunction;
    this.derivativeOfActivationFunction = derivativeOfActivationFunction;
    this.shape = shape;
    if (neurons && neurons.length !== 0) {
      this.neurons = neurons as Neuron[];
    } else {
      for (let i = 0; i < quant; i++) {
        this.neurons.push(new Neuron(shape));
      }
    }
  }

  feedforward(inputs: numberArray) {
    let results: numberArray = [];
    for (let neuron of this.neurons) {
      results.push(this.derivativeOfActivationFunction && this.activationFunction ? this.activationFunction(neuron.feedforward(inputs)) : neuron.feedforward(inputs));
    }
    this.lastOut = results;
    return results
  }

  backward(inputs: numberArray, derivatives: numberArray) {
    let lastWeights: numberArray[] = [];
    let newDerivatives: numberArray = [];
    let derivativesOfActivationFunction: numberArray = [];
    if (this.derivativeOfActivationFunction && this.activationFunction) {
      for (let out of this.lastOut) {
        derivativesOfActivationFunction.push(this.derivativeOfActivationFunction(out));
      }
    }

    let d = sum(derivatives);
    for (let i = 0; i < this.neurons.length; i++) {
      let nd = d;
      if (derivativesOfActivationFunction.length !== 0) {
        nd *= derivativesOfActivationFunction[i];
      }
      lastWeights.push(this.neurons[i].backward(inputs, nd));
    }

    let tempDerivatives: numberArray[] = []
    for (let i = 0; i < this.neurons.length; i++) {
      tempDerivatives[i] = [];
      for (let j = 0; j < lastWeights[i].length; j++) {
        tempDerivatives[i][j] = (lastWeights[i][j] * d);
      }
    }

    for (let i = 0; i < tempDerivatives.length; i++) {
      newDerivatives.push(sum(tempDerivatives[i]));
    }

    return newDerivatives
  }

  snapshot() {
    let res: anyObject = { shape: this.shape, neurons: [] };
    if (this.activationFunction && this.derivativeOfActivationFunction) {
      res.activationFunction = this.activationFunction.toString();
      res.derivativeOfActivationFunction = this.derivativeOfActivationFunction.toString();
    }
    for (let neuron of this.neurons) {
      res.neurons.push(neuron.snapshot());
    }
    return res
  }
}

export default Layer;