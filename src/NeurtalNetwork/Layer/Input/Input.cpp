#include "Input.h"

#include <iostream>
#include <random>

InputLayer::InputLayer() : Layer()
{
}

InputLayer::InputLayer(LayerParams params) : Layer(params) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < numNeurons; i++) {
        weights.push_back(dis(gen));
    }

    CLProgram::writeBuffer(bufferKeys["weights"], 0, weights);
}

InputLayer::~InputLayer()
{
}

std::vector<float> InputLayer::predict(std::vector<float> inputs)
{
    neuronValues = inputs;
    // Copy the value of input1 and input2 to the buffer
    CLProgram::writeBuffer(bufferKeys["inputs"], 0, neuronValues);
    forwardPass();
    std::vector<std::vector<float>> networkValues = returnNetworkValues();
    return networkValues[networkValues.size() - 1];
}

void InputLayer::forwardPass()
{
    // Queue our forward pass
    CLProgram::queueKernel(kernelKeys["forward_pass"]);
    CLProgram::queueKernel(kernelKeys["activate"]);

    nextLayer->forwardPass();
}

std::vector<std::vector<float>> InputLayer::returnNetworkValues()
{
    std::vector<std::vector<float>> returnValues;
    returnValues.push_back(neuronValues);
    std::vector<std::vector<float>> nextLayerValues = nextLayer->returnNetworkValues();
    returnValues.insert(returnValues.end(), nextLayerValues.begin(), nextLayerValues.end());
    return returnValues;
}

void InputLayer::learn(std::vector<float> inputs, std::vector<float> correctOutputs, bool printEpoch) {
    std::vector<float> predictedOutput = predict(inputs);

    CLProgram::writeBuffer(bufferKeys["correctOutput"], 0, correctOutputs);
    CLProgram::queueKernel(kernelKeys["learn"]);

    if (printEpoch) {
        std::cout << "    Testing (" << inputs[0] << ", " << inputs[1] << "): { Output: " << predictedOutput[0] <<", Expected: " << correctOutputs[0] << ", Error: " << predictedOutput[0] - correctOutputs[0] << " }" << std::endl;
    }
}

void InputLayer::assignNextLayers(Layer* nextL)
{
    nextLayer = nextL;
}
