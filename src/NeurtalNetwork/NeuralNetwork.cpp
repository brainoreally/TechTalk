#include "NeuralNetwork.h"

#include <iostream>
#include <sstream>
#include <fstream>

#include <thread>

NeuralNetwork::NeuralNetwork() {
}

NeuralNetwork::NeuralNetwork(NetworkParams params) : parameters(params) {
    inputLayer = InputLayer(parameters.inputLayerParams);
    outputLayer = OutputLayer(parameters.outputLayerParams, &inputLayer);
    
    inputLayer.assignNextLayers(&outputLayer);
    training = false;
    earlyEnd = false;
}

NeuralNetwork::~NeuralNetwork()
{
    if (training)
    {
        earlyEnd = true;
        while (training) {
            std::cout << "waiting on training thread to complete..." << std::endl;
        }
    }
    CLProgram::cleanup();
}

void NeuralNetwork::learn(std::vector<std::vector<std::vector<float>>> trainingData)
{
    if (epoch < 1)
        epoch = 1;

    while (cyclesLeft > 0 && !earlyEnd) {
        bool printEpoch = (cyclesLeft % epoch) == 0;
        if (printEpoch)
            std::cout << "Remaining steps " << cyclesLeft << ":" << std::endl;

        --cyclesLeft;
        for(std::vector<std::vector<float>> trainingSet : trainingData)
            inputLayer.learn(trainingSet[0], trainingSet[1], printEpoch);
    }

    training = false;
}

void NeuralNetwork::train(std::vector<std::vector<std::vector<float>>> trainingData, uint32_t cycles, uint32_t epoc) {
    cyclesLeft = cycles;
    epoch = epoc;

    if (!training && cyclesLeft > 0) {
        training = true;
        std::thread training_thread(&NeuralNetwork::learn, this, trainingData);
        training_thread.detach();
    }
}

void NeuralNetwork::predict(std::vector<float> inputs) {
    std::vector<float> outputP = inputLayer.predict(inputs);
    std::cout << "Output for values (" + std::to_string(inputs[0]) + ", " + std::to_string(inputs[1]) + ") is: " + std::to_string(outputP[0]) << std::endl;
}

std::vector<std::vector<float>> NeuralNetwork::returnNetworkValues() {
    return inputLayer.returnNetworkValues();
}
