import Neuron from "./neuron";
import ReLU from "../../tools/ReLU";
import diffReLU from "../../tools/diffReLU";
import sigmoid from "../../tools/sigmoid";
import diffSigmoid from "../../tools/diffSigmoid";

class Layer {
  private neurons: Neuron[] = [];
  private activationFunction: Function | void = undefined
  private derivativeOfActivationFunction: Function | void = undefined;
  private lastOut: numberArray = [];
  private shape: number = 0;
  private activationFunctionName: string | void = undefined;
  private activations: { [key: string]: { function: Function | undefined, derivative: Function | undefined } } = {
    'relu': { function: ReLU, derivative: diffReLU },
    'sigmoid': { function: sigmoid, derivative: diffSigmoid },
    'none': { function: undefined, derivative: undefined }
  };

  constructor(shape: number, quant: number, activationFunction: 'none' | 'relu' | 'sigmoid' = 'none', neurons?: Neuron[]) {
    this.activationFunctionName = activationFunction;
    this.activationFunction = this.activations[activationFunction].function;
    this.derivativeOfActivationFunction = this.activations[activationFunction].derivative;
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

  backward(inputs: numberArray, derivatives: numberArray[]) {
    let lastWeights: numberArray[] = [];
    let newDerivatives: numberArray[] = [];
    let derivativesOfActivationFunction: numberArray = [];

    // compute derivative of activation function, if have activation function
    if (this.derivativeOfActivationFunction && this.activationFunction) {
      for (let out of this.lastOut) {
        derivativesOfActivationFunction.push(this.derivativeOfActivationFunction(out));
      }
    }

    // feedback to neurons
    for (let i = 0; i < this.neurons.length; i++) {
      let d = 0;
      for (let j = 0; j < derivatives.length; j++) {
        d += derivatives[j][i];
      }
      if (derivativesOfActivationFunction.length !== 0) {
        d *= derivativesOfActivationFunction[i];
      }
      lastWeights.push(this.neurons[i].backward(inputs, d));
    }

    // compute next derivatives
    for (let i = 0; i < lastWeights.length; i++) {
      newDerivatives[i] = [];
      for (let j = 0; j < lastWeights[i].length; j++) {
        let d = 0;
        for (let k = 0; k < derivatives.length; k++) {
          d += derivatives[k][i];
        }
        if (derivativesOfActivationFunction.length !== 0) {
          d *= derivativesOfActivationFunction[i];
        }
        newDerivatives[i][j] = d * lastWeights[i][j];
      }
    }

    return newDerivatives
  }

  snapshot() {
    let res: anyObject = { shape: this.shape, neurons: [] };
    res.activationFunction = this.activationFunctionName;
    for (let neuron of this.neurons) {
      res.neurons.push(neuron.snapshot());
    }
    return res
  }
}

export default Layer;