#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork() {
    inputLayer[0] = Neuron();
    inputLayer[1] = Neuron();
    outputLayer[0] = Neuron();
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

void NeuralNetwork::train(GLuint cycles) {
    for (int i = 0; i < cycles; i++) {
        outputLayer[0].learn(1.0f, 1.0f, 1.0f); // True  or True  = True
        outputLayer[0].learn(1.0f, 0.0f, 1.0f); // True  or False = True
        outputLayer[0].learn(0.0f, 1.0f, 1.0f); // False or True  = True
        outputLayer[0].learn(0.0f, 0.0f, 1.0f); // False or False = False
    }
}

void NeuralNetwork::predict(GLfloat input1, GLfloat input2) {
    // Copy the value of input1 and input2 to the buffer
    err = clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0, sizeof(GLfloat), &input1, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, sizeof(GLfloat), sizeof(GLfloat), &input2, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, weightBuffer, CL_TRUE, 0, 3 * sizeof(GLfloat), &outputLayer[0].weights, 0, NULL, NULL);

    GLfloat xOffset;

    size_t global_size[1] = { 1 };
    size_t local_size[1] = { 1 };

    // Execute the kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_size, local_size, 0, NULL, NULL);

    GLfloat outputP;

    // Copy the output from the buffer
    err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, sizeof(float), &outputP, 0, NULL, NULL);

    std::cout << "Output for values (" + std::to_string(input1) + ", " + std::to_string(input2) + ") is: " + std::to_string(outputP) << std::endl;
}

void NeuralNetwork::initCL()
{
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);

    // Compile and load the kernel
    const char* kernel_source =
        "float heavySideActivation(float neuronOutput)\n"
        "{\n"
        "    if (neuronOutput > 0.0f)\n"
        "        neuronOutput = 1.00f;\n"
        "    else\n"
        "        neuronOutput = 0.0f;\n"
        "\n"
        "    return neuronOutput;\n"
        "}\n"
        "__kernel void forward_pass(__global float* inputs, __global float* weights, __global float* output)\n"
        "{\n"
        "    float bias = 1.0f;\n"
        "    output[0] = heavySideActivation((inputs[0] * weights[0]) + (inputs[1] * weights[1]) + (bias * weights[2]));\n"
        "}\n";

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
