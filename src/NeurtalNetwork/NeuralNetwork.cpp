#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork() {
    CLProgram::initCL();
    inputLayer = Layer(2);
    outputLayer = Layer(1);
    Neuron::setupBuffers();
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

        while (cyclesLeft > 0 && !earlyEnd) {
            bool printEpoch = (cyclesLeft % epoch) == 0;
            if (printEpoch)
                std::cout << "Remaining steps " << cyclesLeft << ":" << std::endl;

            --cyclesLeft;
            inputLayer.learn(1.0f, 1.0f, 1.0f, printEpoch); // True  or True  = True
            inputLayer.learn(1.0f, 0.0f, 1.0f, printEpoch); // True  or False = True
            inputLayer.learn(0.0f, 1.0f, 1.0f, printEpoch); // False or True  = True
            inputLayer.learn(0.0f, 0.0f, 0.0f, printEpoch); // False or False = False
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

    draw();
}

void NeuralNetwork::draw()
{
    inputLayer.draw(glm::vec3(-0.75f, -0.75f, -5.0f));
    outputLayer.draw(glm::vec3(0.75f, 0.00f, -5.0f));
}

void NeuralNetwork::train(GLuint cycles, GLuint epoc) {
    cyclesLeft = cycles;
    epoch = epoc;
}

void NeuralNetwork::predict(GLfloat input1, GLfloat input2) {
    GLfloat outputP = inputLayer.forwardPass(input1, input2);
    outputLayer.setNodeValues({ outputP });
    std::cout << "Output for values (" + std::to_string(input1) + ", " + std::to_string(input2) + ") is: " + std::to_string(outputP) << std::endl;
}

