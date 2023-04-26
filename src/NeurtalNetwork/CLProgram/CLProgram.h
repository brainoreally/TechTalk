#pragma once

#include <CL/cl.h>
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

template<int global_size, int local_size>
struct KernelMap {
    cl_kernel id;

    size_t global_size[global_size];
    size_t local_size[local_size];
};

class CLProgram {
public:
    static void cleanup();

    static void initCL();

    static unsigned int writeBuffer(const char* buffer_key, unsigned int offset, std::vector<float> data);
    static unsigned int writeBuffer(const char* buffer_key, unsigned int offset, float data);

    static float readBuffer(const char* buffer_key, int offset, int size);

    static void queueKernel(const char* kernel_key, bool wait_for_event = true);

private:
    static cl_int err;
    static cl_platform_id platform_id;
    static cl_device_id device_id;
    static cl_context context;
    static cl_command_queue command_queue;
    static cl_program program;

    static std::map<const char*, KernelMap<1, 1>> kernels;
    static std::map<const char*, cl_mem> buffers;

    static std::string loadKernelSource(const char* filename);
};