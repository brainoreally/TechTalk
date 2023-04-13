#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork() {
    weights[0] = 0.2f;
    weights[1] = 0.3f;
    weights[2] = 0.5f;
}

NeuralNetwork::~NeuralNetwork()
{
}

void NeuralNetwork::train(GLuint cycles) {
    for (int i = 0; i < cycles; i++) {
        learn(1.0f, 1.0f, 1.0f); // True  or True  = True
        learn(1.0f, 0.0f, 1.0f); // True  or False = True
        learn(0.0f, 1.0f, 1.0f); // False or True  = True
        learn(0.0f, 0.0f, 1.0f); // False or False = False
    }
}

void NeuralNetwork::learn(GLfloat input1, GLfloat input2, GLfloat output) {
    GLfloat outputP = forwardPass(input1, input2);

    if (outputP > 0.0f)
        outputP = 1.0f;
    else
        outputP = 0.0f;

    GLfloat error = output - outputP;
    weights[0] += error * input1 * learningRate;
    weights[0] += error * input2 * learningRate;
    weights[0] += error * bias * learningRate;
}

GLfloat NeuralNetwork::forwardPass(GLfloat input1, GLfloat input2)
{
    return (input1 * weights[0]) + (input2 * weights[1]) + (bias * weights[2]);
}

void NeuralNetwork::predict(GLfloat input1, GLfloat input2) {
    GLfloat outputP = (input1 * weights[0]) + (input2 * weights[1]) + (bias * weights[2]);
    if (outputP > 0.0f)
        outputP = 1.0f;
    else
        outputP = 0.0f;

    std::cout << "Output for values (" + std::to_string(input1) + ", " + std::to_string(input2) + ") is: " + std::to_string(outputP) << std::endl;
}
