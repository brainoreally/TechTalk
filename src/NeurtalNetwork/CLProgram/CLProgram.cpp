#include "CLProgram.h"

#include <iostream>
#include <sstream>
#include <fstream>

cl_program CLProgram::program = nullptr;
cl_int CLProgram::err = 0;
cl_platform_id CLProgram::platform_id = nullptr;
cl_device_id CLProgram::device_id = nullptr;
cl_context CLProgram::context = nullptr;
cl_command_queue CLProgram::command_queue = nullptr;

std::map<const char*, cl_kernel> CLProgram::kernels = {};
std::map<const char*, cl_mem> CLProgram::buffers = {};

void CLProgram::cleanup()
{
	// Clean up
	for (auto iter = buffers.begin(); iter != buffers.end(); ++iter) {
		clReleaseMemObject(iter->second);
	}
	for (auto iter = kernels.begin(); iter != kernels.end(); ++iter) {
		clReleaseKernel(iter->second);
	}
	clReleaseProgram(program);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
}

void CLProgram::initCL(const char* kernel_source_path)
{
    clGetPlatformIDs(1, &platform_id, NULL);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);

    std::string kernelCode = loadKernelSource(kernel_source_path);
    const char* kernel_source = kernelCode.c_str();

    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    // Build OpenCL program
    cl_int err = clBuildProgram(program, 1, &device_id, "-cl-fast-relaxed-math", NULL, NULL);
    if (err == CL_INVALID_BUILD_OPTIONS) {
        std::cerr << "Error building OpenCL program: " << err << " - Invalid build options" << std::endl;
        // Handle the error appropriately
        // ...
    }
    else if (err == CL_BUILD_PROGRAM_FAILURE) {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char* log = (char*)malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
    }
    else if (err != CL_SUCCESS) {
        std::cerr << "Error building OpenCL program: " << err << std::endl;
        // Handle the error appropriately
        // ...
    }
 }

void CLProgram::setKernelParam(const char* kernel_key, int param_order, const char* buffer_key)
{
    err = clSetKernelArg(kernels[kernel_key], param_order, sizeof(cl_mem), &buffers[buffer_key]);
}

void CLProgram::createKernel(const char* kernel_key)
{
    kernels[kernel_key] = clCreateKernel(program, kernel_key, &err);
}

void CLProgram::queueKernel(const char* kernel_key, size_t global, size_t local)
{
    cl_event kernel_event;

    if (err != 0)
        int test = 1;
    // Execute the kernel
    err = clEnqueueNDRangeKernel(command_queue, kernels[kernel_key], 1, NULL, &global, &local, 0, NULL, &kernel_event);
    if (err != 0)
        int test = 1;
    // Wait for the kernel to complete
    clWaitForEvents(1, &kernel_event);

    // Release the event object
    clReleaseEvent(kernel_event);
}

std::string CLProgram::loadKernelSource(const char* filename)
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

void CLProgram::setupNetworkOpenCL(NetworkParams* params) {
    createBuffer<unsigned int>("networkCounts", 8); // [0: numNeuronsPerSample, 1: numWeightsPerSample, 2: numLayers, 3: inputLayerSize, , 4: outputLayerSize, 5: numSamples, 6: trainingLeft, 7: epoch]
    createBuffer<unsigned int>("layerSizes", params->numLayers); // sizes (#neurons) of each weighted layer
    createBuffer<unsigned int>("layerActivations", params->numLayers); // enum for choosing layer activation method (0 = sigmod, 1 = relu)
    createBuffer<float>("correctOutput", params->numOutputs * params->numSamples); // float collection - #neurons in network + biases - store all network/layer/node values
    createBuffer<float>("neuronValues", params->numNeurons * params->numSamples); // float collection - #neurons in network + biases - store all network/layer/node values
    createBuffer<float>("weights", params->numWeights); // float collection - #values with a weight (inputs on layers + bias) - store all weights in network/layers
    createBuffer<float>("biases", params->numNeurons); // float collection - #bias param for each neuron
    createBuffer<float>("weightDerivitiveOut", params->numNeurons * params->numSamples); // float collection - #values with a weight (inputs on layers + bias) - store all weights in network/layers

    std::vector<KernelParam> network_kernels = {
        { "network_output", { "networkCounts", "layerSizes", "correctOutput", "neuronValues" }},
        { "forward_pass", { "networkCounts", "layerSizes", "layerActivations", "neuronValues", "weights", "biases" }},
        { "backward_pass", { "networkCounts", "layerSizes", "layerActivations", "correctOutput", "neuronValues", "weights", "biases", "weightDerivitiveOut" }},
    };
    for (KernelParam param : network_kernels)
    {
        createKernel(param.key);
        for (int i = 0; i < param.param_buffer_keys.size(); i++)
            setKernelParam(param.key, i, param.param_buffer_keys[i]);
    }
}