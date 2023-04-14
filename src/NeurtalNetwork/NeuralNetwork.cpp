#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork() {
    inputLayer[0] = Neuron(&queue, &kernel, &outputBuffer, &inputBuffer, &weightBuffer);
    inputLayer[1] = Neuron(&queue, &kernel, &outputBuffer, &inputBuffer, &weightBuffer);
    outputLayer[0] = Neuron(&queue, &kernel, &outputBuffer, &inputBuffer, &weightBuffer);
    initCL();
 }

NeuralNetwork::~NeuralNetwork()
{
    // Clean up
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseMemObject(weightBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

void NeuralNetwork::draw()
{
    inputLayer[0].draw(glm::vec3(-0.75f, 0.75f, -5.0f));
    inputLayer[1].draw(glm::vec3(-0.75f, -0.75f, -5.0f));
    outputLayer[0].draw(glm::vec3(0.75f, 0.0f, -5.0f));
}

void NeuralNetwork::train(GLuint cycles, GLuint epoch) {
    for (int i = 0; i < cycles; i++) {
        bool printEpoch = (i % epoch) == 0;
        if (printEpoch)
            std::cout << "Iteration " << i << ":" << std::endl;
        outputLayer[0].learn(1.0f, 1.0f, 1.0f, printEpoch); // True  or True  = True
        outputLayer[0].learn(1.0f, 0.0f, 1.0f, printEpoch); // True  or False = True
        outputLayer[0].learn(0.0f, 1.0f, 1.0f, printEpoch); // False or True  = True
        outputLayer[0].learn(0.0f, 0.0f, 0.0f, printEpoch); // False or False = False
    }
}

void NeuralNetwork::predict(GLfloat input1, GLfloat input2) {
    GLfloat outputP = outputLayer[0].forwardPass(input1, input2);
    inputLayer[0].changeColour(input1);
    inputLayer[1].changeColour(input2);
    outputLayer[0].changeColour(outputP);
    std::cout << "Output for values (" + std::to_string(input1) + ", " + std::to_string(input2) + ") is: " + std::to_string(outputP) << std::endl;
}

void NeuralNetwork::initCL()
{
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);


    std::string kernelCode = loadKernelSource("src/kernels/forwardPass.cl");
    const GLchar* kernel_source = kernelCode.c_str();

    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "forward_pass", &err);

    // Create a buffer to hold the output
    inputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * sizeof(GLfloat), NULL, &err);
    weightBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 3 * sizeof(GLfloat), NULL, &err);
    outputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(GLfloat), NULL, &err);

    // Set the argument of the kernel to the buffer
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &weightBuffer);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputBuffer);
 }

std::string NeuralNetwork::loadKernelSource(const char* filename)
{
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return "";
    }

    std::stringstream buffer;
    buffer << file.rdbuf();

    file.close();

    return buffer.str();
}
