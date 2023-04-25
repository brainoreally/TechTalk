#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork() {
    initCL();
    inputLayer = Layer(2, clProgramInf);
    outputLayer = Layer(1, clProgramInf);
    Neuron::setupBuffers();
    training = false;
 }

NeuralNetwork::~NeuralNetwork()
{
    // Clean up
    clReleaseMemObject(clProgramInf.inputBuffer);
    clReleaseMemObject(clProgramInf.outputBuffer);
    clReleaseMemObject(clProgramInf.weightBuffer);
    clReleaseMemObject(clProgramInf.correctOutputBuffer);
    clReleaseKernel(clProgramInf.forwardPassKernel);
    clReleaseKernel(clProgramInf.activateKernel);
    clReleaseProgram(clProgramInf.program);
    clReleaseCommandQueue(clProgramInf.queue);
    clReleaseContext(clProgramInf.context);
}

void NeuralNetwork::learn()
{
        if (epoch < 1)
            epoch = 1;

        while (cyclesLeft > 0) {
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

void NeuralNetwork::initCL()
{
    clGetPlatformIDs(1, &clProgramInf.platform, NULL);
    clGetDeviceIDs(clProgramInf.platform, CL_DEVICE_TYPE_GPU, 1, &clProgramInf.device, NULL);
    clProgramInf.context = clCreateContext(NULL, 1, &clProgramInf.device, NULL, NULL, &clProgramInf.err);
    clProgramInf.queue = clCreateCommandQueueWithProperties(clProgramInf.context, clProgramInf.device, 0, &clProgramInf.err);

    std::string kernelCode = loadKernelSource("src/kernels/forwardPass.cl");
    const GLchar* kernel_source = kernelCode.c_str();

    clProgramInf.program = clCreateProgramWithSource(clProgramInf.context, 1, &kernel_source, NULL, &clProgramInf.err);
    // Build OpenCL program
    cl_int err = clBuildProgram(clProgramInf.program, 1, &clProgramInf.device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err == CL_INVALID_BUILD_OPTIONS) {
        std::cerr << "Error building OpenCL program: " << err << " - Invalid build options" << std::endl;
        // Handle the error appropriately
        // ...
    }
    else if (err == CL_BUILD_PROGRAM_FAILURE) {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(clProgramInf.program, clProgramInf.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char* log = (char*)malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(clProgramInf.program, clProgramInf.device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
    }
    else if (err != CL_SUCCESS) {
        std::cerr << "Error building OpenCL program: " << err << std::endl;
        // Handle the error appropriately
        // ...
    }
    clProgramInf.forwardPassKernel = clCreateKernel(clProgramInf.program, "forward_pass", &clProgramInf.err);
    clProgramInf.activateKernel = clCreateKernel(clProgramInf.program, "activate", &clProgramInf.err);
    clProgramInf.learnKernel = clCreateKernel(clProgramInf.program, "learn", &clProgramInf.err);

    // Create a buffer to hold the output
    clProgramInf.inputBuffer =  clCreateBuffer(clProgramInf.context, CL_MEM_READ_WRITE, 2 * sizeof(GLfloat), NULL, &clProgramInf.err);
    clProgramInf.weightBuffer = clCreateBuffer(clProgramInf.context, CL_MEM_READ_WRITE, 3 * sizeof(GLfloat), NULL, &clProgramInf.err);
    clProgramInf.outputBuffer = clCreateBuffer(clProgramInf.context, CL_MEM_READ_WRITE, sizeof(GLfloat), NULL, &clProgramInf.err);
    clProgramInf.correctOutputBuffer = clCreateBuffer(clProgramInf.context, CL_MEM_READ_WRITE, sizeof(GLfloat), NULL, &clProgramInf.err);

    // Set the argument of the kernel to the buffer
    clProgramInf.err = clSetKernelArg(clProgramInf.forwardPassKernel, 0, sizeof(cl_mem), &clProgramInf.inputBuffer);
    clProgramInf.err = clSetKernelArg(clProgramInf.forwardPassKernel, 1, sizeof(cl_mem), &clProgramInf.weightBuffer);

    clProgramInf.err = clSetKernelArg(clProgramInf.activateKernel, 0, sizeof(cl_mem), &clProgramInf.inputBuffer);
    clProgramInf.err = clSetKernelArg(clProgramInf.activateKernel, 1, sizeof(cl_mem), &clProgramInf.weightBuffer);
    clProgramInf.err = clSetKernelArg(clProgramInf.activateKernel, 2, sizeof(cl_mem), &clProgramInf.outputBuffer);

    clProgramInf.err = clSetKernelArg(clProgramInf.learnKernel, 0, sizeof(cl_mem), &clProgramInf.inputBuffer);
    clProgramInf.err = clSetKernelArg(clProgramInf.learnKernel, 1, sizeof(cl_mem), &clProgramInf.weightBuffer);
    clProgramInf.err = clSetKernelArg(clProgramInf.learnKernel, 2, sizeof(cl_mem), &clProgramInf.outputBuffer);
    clProgramInf.err = clSetKernelArg(clProgramInf.learnKernel, 3, sizeof(cl_mem), &clProgramInf.correctOutputBuffer);

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
