#include "Input.h"

#include <iostream>
#include <random>

InputLayer::InputLayer() : Layer()
{
}

InputLayer::InputLayer(Layer* next, int numNeuron) : Layer(numNeuron) {
    nextLayer = next;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < numNeurons; i++) {
        weights.push_back(dis(gen));
    }

    CLProgram::writeBuffer("weights", 0, weights);
}

InputLayer::~InputLayer()
{
}

std::vector<float> InputLayer::forwardPass(std::vector<float> inputs)
{
    neuronValues = inputs;

    // Copy the value of input1 and input2 to the buffer
    CLProgram::writeBuffer("inputs", 0, neuronValues);
    // Queue our forward pass
    CLProgram::queueKernel("forward_pass");

    // Return the activated value
    return nextLayer->forwardPass(activate());
}

std::vector<std::vector<float>> InputLayer::returnNetworkValues()
{
    std::vector<std::vector<float>> returnValues;
    returnValues.push_back(neuronValues);
    std::vector<std::vector<float>> nextLayerValues = nextLayer->returnNetworkValues();
    returnValues.insert(returnValues.end(), nextLayerValues.begin(), nextLayerValues.end());
    return returnValues;
}

std::vector<float> InputLayer::activate()
{
    CLProgram::queueKernel("activate");

    std::vector<float> outputP = CLProgram::readBuffer("output", 0, 1);
    
    return outputP;
}

void InputLayer::learn(std::vector<float> inputs, std::vector<float> outputs, bool printEpoch) {
    std::vector<float> outputP = forwardPass(inputs);

    CLProgram::writeBuffer("correctOutput", 0, outputs);
    CLProgram::queueKernel("learn");

    if (printEpoch) {
        std::cout << "    Testing (" << inputs[0] << ", " << inputs[1] << "): { Output: " << outputP[0] <<", Expected: " << outputs[0] << ", Error: " << outputP[0] - outputs[0] << " }" << std::endl;
    }
}