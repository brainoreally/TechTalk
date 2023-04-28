#include "Output.h"
#include <random>

OutputLayer::OutputLayer() : Layer(), previousLayer(nullptr)
{
}

OutputLayer::OutputLayer(LayerParams params, Layer* prevLayer) : Layer(params), previousLayer(prevLayer) 
{
}

OutputLayer::~OutputLayer()
{
}

void OutputLayer::forwardPass()
{
    neuronValues = CLProgram::readBuffer(bufferKeys["output"], 0, numNeurons);
}

std::vector<std::vector<float>> OutputLayer::returnNetworkValues()
{
    std::vector<std::vector<float>> returnValues;
    returnValues.push_back(neuronValues);
    return returnValues;
}
