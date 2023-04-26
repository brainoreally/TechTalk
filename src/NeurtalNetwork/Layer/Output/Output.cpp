#include "Output.h"
#include <random>

OutputLayer::OutputLayer()
{
}

OutputLayer::OutputLayer(LayerParams params, Layer* previousLayer) : Layer(params) {
}

OutputLayer::~OutputLayer()
{
}

void OutputLayer::forwardPass()
{
    neuronValues = CLProgram::readBuffer(bufferKeys["output"], 0, 1);
}

std::vector<std::vector<float>> OutputLayer::returnNetworkValues()
{
    std::vector<std::vector<float>> returnValues;
    returnValues.push_back(neuronValues);
    return returnValues;
}
