#include "Output.h"
#include <random>

OutputLayer::OutputLayer()
{
}

OutputLayer::OutputLayer(int numNeuron) : Layer(numNeuron) {

}

OutputLayer::~OutputLayer()
{
}

std::vector<float> OutputLayer::forwardPass(std::vector<float> inputs)
{
    neuronValues = inputs;

    return neuronValues;
}

std::vector<std::vector<float>> OutputLayer::returnNetworkValues()
{
    std::vector<std::vector<float>> returnValues;
    returnValues.push_back(neuronValues);
    return returnValues;
}
