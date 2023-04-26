#include "CLProgram.h"

cl_program CLProgram::program = nullptr;
cl_int CLProgram::err = 0;
cl_platform_id CLProgram::platform_id = nullptr;
cl_device_id CLProgram::device_id = nullptr;
cl_context CLProgram::context = nullptr;
cl_command_queue CLProgram::command_queue = nullptr;

std::map<const char*, KernelMap<1, 1>> CLProgram::kernels = {};
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

void CLProgram::initCL()
{
    clGetPlatformIDs(1, &platform_id, NULL);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);

    std::string kernelCode = loadKernelSource("src/kernels/perceptronNetwork.cl");
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

    kernels["forward_pass"].id = clCreateKernel(program, "forward_pass", &err); 
    kernels["forward_pass"].global_size[0] = 2;
    kernels["forward_pass"].local_size[0] = 1;

    kernels["activate"].id = clCreateKernel(program, "activate", &err);
    kernels["activate"].global_size[0] = 1;
    kernels["activate"].local_size[0] = 1;

    kernels["learn"].id = clCreateKernel(program, "learn", &err);
    kernels["learn"].global_size[0] = 3;
    kernels["learn"].local_size[0] = 1;

    // Create a buffer to hold the output
    buffers["inputs"] = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * sizeof(float), NULL, &err);
    buffers["weights"] = clCreateBuffer(context, CL_MEM_READ_WRITE, 3 * sizeof(float), NULL, &err);
    buffers["output"] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, &err);
    buffers["correctOutput"] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, &err);

    // Set the argument of the kernel to the buffer
    err = clSetKernelArg(kernels["forward_pass"].id, 0, sizeof(cl_mem), &buffers["inputs"]);
    err = clSetKernelArg(kernels["forward_pass"].id, 1, sizeof(cl_mem), &buffers["weights"]);

    err = clSetKernelArg(kernels["activate"].id, 0, sizeof(cl_mem), &buffers["inputs"]);
    err = clSetKernelArg(kernels["activate"].id, 1, sizeof(cl_mem), &buffers["weights"]);
    err = clSetKernelArg(kernels["activate"].id, 2, sizeof(cl_mem), &buffers["output"]);

    err = clSetKernelArg(kernels["learn"].id, 0, sizeof(cl_mem), &buffers["inputs"]);
    err = clSetKernelArg(kernels["learn"].id, 1, sizeof(cl_mem), &buffers["weights"]);
    err = clSetKernelArg(kernels["learn"].id, 2, sizeof(cl_mem), &buffers["output"]);
    err = clSetKernelArg(kernels["learn"].id, 3, sizeof(cl_mem), &buffers["correctOutput"]);
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

float CLProgram::readBuffer(const char* buffer_key, int offset, int size)
{
    float output;
    err = clEnqueueReadBuffer(command_queue, buffers[buffer_key], CL_TRUE, offset, size * sizeof(float), &output, 0, NULL, NULL);
    return output;
}

void CLProgram::queueKernel(const char* kernel_key, bool wait_for_event)
{
    // Execute the kernel
    cl_event event;
    err = clEnqueueNDRangeKernel(command_queue, kernels[kernel_key].id, 1, NULL, kernels[kernel_key].global_size, kernels[kernel_key].local_size, 0, NULL, &event);
        
    if(wait_for_event)
        clWaitForEvents(1, &event);
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