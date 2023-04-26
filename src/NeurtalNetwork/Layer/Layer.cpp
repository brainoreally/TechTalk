#include "Layer.h"

Layer::Layer()
{
}

Layer::Layer(int numNeuron) {
    weights.push_back(0.2f);
    weights.push_back(0.3f);
    weights.push_back(0.5f);

    CLProgram::writeBuffer("weights", 0, weights);
    numNeurons = numNeuron;

    for (int i = 0; i < numNeurons; i++) {
        nodeValues.push_back(0.0f);
    }
}

Layer::~Layer()
{
}

void Layer::draw(glm::vec3 position)
{
    // Draw Neurons
    GLfloat offset = 0.0f;
    for (int i = 0; i < numNeurons; i++) {
        Neuron::draw(glm::vec3(position.x, position.y + offset, position.z), nodeValues[i]);
        offset += 1.5f;
    }
}

void Layer::setNodeValues(std::vector<GLfloat> newValues) {
    nodeValues = newValues;
}

GLfloat Layer::forwardPass(GLfloat input1, GLfloat input2)
{
    nodeValues[0] = input1;
    nodeValues[1] = input2;

    // Copy the value of input1 and input2 to the buffer
    CLProgram::writeBuffer("inputs", 0, nodeValues);
    // Queue our forward pass
    CLProgram::queueKernel("forward_pass");

    // Return the activated value
    return activate();
}

GLfloat Layer::activate()
{
    CLProgram::queueKernel("activate");

    GLfloat outputP = CLProgram::readBuffer("output", 0, 1);
    
    return outputP;
}

void Layer::learn(GLfloat input1, GLfloat input2, GLfloat output, bool printEpoch) {
    GLfloat outputP = forwardPass(input1, input2);

    CLProgram::writeBuffer("correctOutput", 0, output);
    CLProgram::queueKernel("learn");

    if (printEpoch) {
        std::cout << "    Testing (" << input1 << ", " << input2 << "): { Output: " << outputP <<", Expected: " << output << ", Error: " << outputP - output << " }" << std::endl;
    }
}