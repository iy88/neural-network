import shuffle from "../../tools/shuffle";
import sum from "../../tools/sum";
import Layer from "./layer";
import Neuron from "./neuron";

class Network {
  private layers: Layer[] = [];

  constructor(layers?: Layer[]) {
    layers ? layers.length !== 0 ? this.layers = layers : '' : ''
  }

  load(model: anyObject[]) {
    for (let layer of model) {
      let neurons: Neuron[] = [];
      let activationFunction: undefined | Function = undefined;
      let derivativeOfActivationFunction: undefined | Function = undefined;
      for (let neuron of layer.neurons) {
        neurons.push(new Neuron(0, [...neuron.weigths, neuron.bias]));
      }
      if (layer.activationFunction) {
        activationFunction = eval('(()=>{return ' + layer.activationFunction + '})()');
        derivativeOfActivationFunction = eval('(()=>{return ' + layer.derivativeOfActivationFunction + '})()');
      }
      this.addLayer(layer.shape, neurons.length, activationFunction, derivativeOfActivationFunction, neurons);
    }
  }

  addLayer(shape: number, quant: number, activationFunction?: Function, derivativeOfActivationFunction?: Function, neurons?: Neuron[]) {
    this.layers.push(new Layer(shape, quant, activationFunction, derivativeOfActivationFunction, neurons));
  }

  removeLayer(index?: number) {
    if (index) {
      this.layers.splice(index, 1);
    } else {
      this.layers = [];
    }
  }

  feedforward(inputs: numberArray) {
    let tempResult: numberArray = inputs;
    for (let layer of this.layers) {
      tempResult = layer.feedforward(tempResult);
    }
    return tempResult
  }

  train(trainingData: { input: numberArray, output: numberArray }[], learningRate: number, trainingTime: number) {
    let trainingDataCopy: { input: numberArray, output: numberArray }[] = trainingData.slice(0);
    for (let epoch = 1; epoch <= trainingTime; epoch++) {
      // using SGD
      for (let data of trainingDataCopy) {
        let inputs: numberArray = data.input;
        let y: numberArray = data.output;
        let eachLayerOutPut: numberArray[] = [this.layers[0].feedforward(inputs)];
        for (let i = 1; i < this.layers.length; i++) {
          eachLayerOutPut.push(this.layers[i].feedforward(eachLayerOutPut[eachLayerOutPut.length - 1]));
        }
        // computing partial derivative
        let partialDerivative: numberArray = [];
        for (let i = 0; i < y.length; i++) {
          partialDerivative.push(learningRate * (eachLayerOutPut[eachLayerOutPut.length - 1][i] - y[i]))
        }
        for (let i = this.layers.length - 1; i > -1; i--) {
          partialDerivative = this.layers[i].backward(eachLayerOutPut[i - 1] || inputs, partialDerivative);
        }
      }
      trainingDataCopy = shuffle(trainingDataCopy);
    }
  }

  fit(trainingData: { input: numberArray, output: numberArray }[], testingData: { input: numberArray, output: numberArray }[], learningRate: number, patient: number) {
    let trainingDataCopy: { input: numberArray, output: numberArray }[] = trainingData.slice(0);
    let allEpochLoss: numberArray = [];
    let startPatientCount = false;
    let patientValue = 0;
    for (; ;) {
      // using SGD
      for (let data of trainingDataCopy) {
        let inputs: numberArray = data.input;
        let y: numberArray = data.output;
        let eachLayerOutPut: numberArray[] = [this.layers[0].feedforward(inputs)];
        for (let i = 1; i < this.layers.length; i++) {
          eachLayerOutPut.push(this.layers[i].feedforward(eachLayerOutPut[eachLayerOutPut.length - 1]));
        }
        // computing partial derivative
        let partialDerivative: numberArray = [];
        for (let i = 0; i < y.length; i++) {
          partialDerivative.push(learningRate * (eachLayerOutPut[eachLayerOutPut.length - 1][i] - y[i]))
        }
        for (let i = this.layers.length - 1; i > -1; i--) {
          partialDerivative = this.layers[i].backward(eachLayerOutPut[i - 1] || inputs, partialDerivative);
        }
      }
      let allLoss: numberArray = [];
      for (let data of testingData) {
        let inputs: numberArray = data.input;
        let y: numberArray = data.output;
        let eachLayerOutPut: numberArray[] = [this.layers[0].feedforward(inputs)];
        for (let i = 1; i < this.layers.length; i++) {
          eachLayerOutPut.push(this.layers[i].feedforward(eachLayerOutPut[eachLayerOutPut.length - 1]));
        }
        let partialDerivative: numberArray = [];
        for (let i = 0; i < y.length; i++) {
          partialDerivative.push(0.5 * Math.pow(eachLayerOutPut[eachLayerOutPut.length - 1][i] - y[i], 2));
        }
        let loss = sum(partialDerivative) / partialDerivative.length;
        allLoss.push(loss);
      }
      allEpochLoss.push(sum(allLoss) / allLoss.length);
      if (allEpochLoss[allEpochLoss.length - 1] === 0) {
        break;
      }
      if (allEpochLoss.length > 1 && allEpochLoss[allEpochLoss.length - 1] > allEpochLoss[allEpochLoss.length - 2]) {
        startPatientCount = true;
      }
      if (startPatientCount === true) {
        patientValue++;
        if (patientValue > patient) {
          break;
        }
      }
      trainingDataCopy = shuffle(trainingDataCopy);
    }
    return allEpochLoss
  }

  snapshot() {
    let configs: anyObject[] = [];
    for (let layer of this.layers) {
      configs.push(layer.snapshot());
    }
    return configs;
  }

}

export default Network;