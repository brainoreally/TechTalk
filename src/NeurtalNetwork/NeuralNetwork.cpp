#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork() {
    inputLayer[0] = Neuron();
    inputLayer[1] = Neuron();
    outputLayer[0] = Neuron();
 }

NeuralNetwork::~NeuralNetwork()
{
}

void NeuralNetwork::draw(GLfloat xOffset)
{
    inputLayer[0].draw(glm::vec3(xOffset, 0.75f, -5.0f));
    inputLayer[1].draw(glm::vec3(xOffset, -0.75f, -5.0f));
    outputLayer[0].draw(glm::vec3(xOffset + 2, 0.0f, -5.0f));
}

void NeuralNetwork::train(GLuint cycles) {
    for (int i = 0; i < cycles; i++) {
        outputLayer[0].learn(1.0f, 1.0f, 1.0f); // True  or True  = True
        outputLayer[0].learn(1.0f, 0.0f, 1.0f); // True  or False = True
        outputLayer[0].learn(0.0f, 1.0f, 1.0f); // False or True  = True
        outputLayer[0].learn(0.0f, 0.0f, 1.0f); // False or False = False
    }
}

void NeuralNetwork::predict(GLfloat input1, GLfloat input2) {
    GLfloat outputP = outputLayer[0].forwardPass(input1, input2);
    std::cout << "Output for values (" + std::to_string(input1) + ", " + std::to_string(input2) + ") is: " + std::to_string(outputP) << std::endl;
}
