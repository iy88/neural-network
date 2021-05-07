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

  backward(inputs: numberArray, derivatives: numberArray[]) {
    let lastWeights: numberArray[] = [];
    let newDerivatives: numberArray[] = [];
    let derivativesOfActivationFunction: numberArray = [];
    if (this.derivativeOfActivationFunction && this.activationFunction) {
      for (let out of this.lastOut) {
        derivativesOfActivationFunction.push(this.derivativeOfActivationFunction(out));
      }
    }

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

    let temp: numberArray[] = [];
    for (let i = 0; i < lastWeights.length; i++) {
      temp[i] = [];
      for (let j = 0; j < lastWeights[i].length; j++) {
        let d = 0;
        for (let k = 0; k < derivatives.length; k++) {
          d += derivatives[k][i];
        }
        if (derivativesOfActivationFunction.length !== 0) {
          d *= derivativesOfActivationFunction[i];
        }
        temp[i][j] = d * lastWeights[i][j];
      }
    }

    newDerivatives = temp;

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