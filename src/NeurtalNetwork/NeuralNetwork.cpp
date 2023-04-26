#include "NeuralNetwork.h"

#include <iostream>
#include <sstream>
#include <fstream>

#include <thread>

NeuralNetwork::NeuralNetwork() {
    CLProgram::initCL();
    outputLayer = OutputLayer(1);
    inputLayer = InputLayer(&outputLayer, 2);
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

void NeuralNetwork::learn()
{
        if (epoch < 1)
            epoch = 1;

        std::vector<std::vector<std::vector<float>>> trainingData =
        { 
            {{1.0f, 1.0f}, {1.0f}}, // True  or True  = True
            {{1.0f, 0.0f}, {1.0f}}, // True  or False = True
            {{0.0f, 1.0f}, {1.0f}}, // False or True  = True
            {{0.0f, 0.0f}, {0.0f}}, // False or False = False
        };
        while (cyclesLeft > 0 && !earlyEnd) {
            bool printEpoch = (cyclesLeft % epoch) == 0;
            if (printEpoch)
                std::cout << "Remaining steps " << cyclesLeft << ":" << std::endl;

            --cyclesLeft;
            inputLayer.learn(trainingData[0][0], trainingData[0][1], printEpoch);
            inputLayer.learn(trainingData[1][0], trainingData[1][1], printEpoch);
            inputLayer.learn(trainingData[2][0], trainingData[2][1], printEpoch);
            inputLayer.learn(trainingData[3][0], trainingData[3][1], printEpoch);
        }

        training = false;
}

void NeuralNetwork::loop()
{
    if (!training && cyclesLeft > 0) {
        training = true;

        std::thread training_thread([&]() { learn(); });
        training_thread.detach();
    }
}

void NeuralNetwork::train(uint32_t cycles, uint32_t epoc) {
    cyclesLeft = cycles;
    epoch = epoc;
}

void NeuralNetwork::predict(std::vector<float> inputs) {
    std::vector<float> outputP = inputLayer.forwardPass(inputs);
    std::cout << "Output for values (" + std::to_string(inputs[0]) + ", " + std::to_string(inputs[1]) + ") is: " + std::to_string(outputP[0]) << std::endl;
}

std::vector<std::vector<float>> NeuralNetwork::returnNetworkValues() {
    return inputLayer.returnNetworkValues();
}
