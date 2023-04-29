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

std::map<const char*, KernelParams> CLProgram::kernels = {};
std::map<const char*, cl_mem> CLProgram::buffers = {};

void CLProgram::cleanup()
{
	// Clean up
	for (auto iter = buffers.begin(); iter != buffers.end(); ++iter) {
		clReleaseMemObject(iter->second);
	}
	for (auto iter = kernels.begin(); iter != kernels.end(); ++iter) {
		clReleaseKernel(iter->second.id);
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
    err = clSetKernelArg(kernels[kernel_key].id, param_order, sizeof(cl_mem), &buffers[buffer_key]);
}

void CLProgram::createBuffer(const char* buffer_key, int buffer_size)
{
    buffers[buffer_key] = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size * sizeof(float), NULL, &err);
}

unsigned int CLProgram::writeBuffer(const char* buffer_key, unsigned int offset, std::vector<float> data) {
    clEnqueueWriteBuffer(command_queue, buffers[buffer_key], CL_TRUE, offset, data.size() * sizeof(float), &data[0], 0, NULL, NULL);
    unsigned int newOffset = offset + (data.size() * sizeof(float));
    return newOffset;
}

unsigned int CLProgram::writeBuffer(const char* buffer_key, unsigned int offset, float data) {
    clEnqueueWriteBuffer(command_queue, buffers[buffer_key], CL_TRUE, offset, sizeof(float), &data, 0, NULL, NULL);
    unsigned int newOffset = offset + sizeof(float);
    return newOffset;
}

std::vector<float> CLProgram::readBuffer(const char* buffer_key, int offset, int size)
{
    std::vector<float> output = { 0.0f };
    err = clEnqueueReadBuffer(command_queue, buffers[buffer_key], CL_TRUE, offset, size * sizeof(float), &output[0], 0, NULL, NULL);
    return output;
}

void CLProgram::createKernel(const char* kernel_key, size_t* global, size_t* local)
{
    kernels[kernel_key].id = clCreateKernel(program, kernel_key, &err);
    kernels[kernel_key].global_size = global;
    kernels[kernel_key].local_size = local;
}

void CLProgram::queueKernel(const char* kernel_key)
{
    // Execute the kernel
    err = clEnqueueNDRangeKernel(command_queue, kernels[kernel_key].id, 1, NULL, kernels[kernel_key].global_size, kernels[kernel_key].local_size, 0, NULL, NULL);
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