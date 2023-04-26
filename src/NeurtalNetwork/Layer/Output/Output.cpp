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

void OutputLayer::forwardPass()
{
    neuronValues = CLProgram::readBuffer("output", 0, 1);
}

std::vector<std::vector<float>> OutputLayer::returnNetworkValues()
{
    std::vector<std::vector<float>> returnValues;
    returnValues.push_back(neuronValues);
    return returnValues;
}
