#include "Layer.h"

Layer::Layer()
{
}

Layer::Layer(int numNeuron, CLContainer clProgramInf) {
    Layer::clInf = clProgramInf;

    weights.push_back(0.2f);
    weights.push_back(0.3f);
    weights.push_back(0.5f);

    cl_int err = clEnqueueWriteBuffer(Layer::clInf.queue, Layer::clInf.weightBuffer, CL_TRUE, 0, 3 * sizeof(GLfloat), &weights[0], 0, NULL, NULL);
    numNeurons = numNeuron;

    for (int i = 0; i < numNeurons; i++) {
        nodeValues.push_back(0.0f);
    }
}

Layer::~Layer()
{
}

CLContainer Layer::clInf = CLContainer();

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
    cl_int err;
    err = clEnqueueWriteBuffer(Layer::clInf.queue, Layer::clInf.inputBuffer, CL_TRUE, 0,               sizeof(GLfloat), &input1, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(Layer::clInf.queue, Layer::clInf.inputBuffer, CL_TRUE, sizeof(GLfloat), sizeof(GLfloat), &input2, 0, NULL, NULL);

    size_t global_size[1] = { 2 };
    size_t local_size[1] = { 1 };

    // Execute the kernel
    cl_event event;
    err = clEnqueueNDRangeKernel(Layer::clInf.queue, Layer::clInf.forwardPassKernel, 1, NULL, global_size, local_size, 0, NULL,  &event);
    clWaitForEvents(1, &event);

    return activate();
}

GLfloat Layer::activate()
{
    GLfloat outputP = 0.0f;

    size_t global_size2[1] = { 1 };
    size_t local_size2[1] = { 1 };

    // Execute the kernel
    cl_event event;
    cl_int err = clEnqueueNDRangeKernel(Layer::clInf.queue, Layer::clInf.activateKernel, 1, NULL, global_size2, local_size2, 0, NULL,  &event);
    clWaitForEvents(1, &event);

    err = clEnqueueReadBuffer(Layer::clInf.queue, Layer::clInf.outputBuffer, CL_TRUE, 0, sizeof(float), &outputP, 0, NULL, NULL);
    return outputP;
}

void Layer::learn(GLfloat input1, GLfloat input2, GLfloat output, bool printEpoch) {
    GLfloat outputP = forwardPass(input1, input2);

    size_t global_size[1] = { 3 };
    size_t local_size[1] = { 1 };

    cl_int err = clEnqueueWriteBuffer(Layer::clInf.queue, Layer::clInf.correctOutputBuffer, CL_TRUE, 0, sizeof(GLfloat), &output, 0, NULL, NULL);

    cl_event event;
    // Execute the kernel
    err = clEnqueueNDRangeKernel(Layer::clInf.queue, Layer::clInf.learnKernel, 1, NULL, global_size, local_size, 0, NULL, &event);
    clWaitForEvents(1, &event);

    if (printEpoch) {
        std::cout << "    Testing (" << input1 << ", " << input2 << "): { Output: " << outputP <<", Expected: " << output << ", Error: " << outputP - output << " }" << std::endl;
    }
}