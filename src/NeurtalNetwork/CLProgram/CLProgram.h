#pragma once

#include <CL/cl.h>
#include <vector>
#include <map>
#include <string>

class CLProgram {
public:
    static void cleanup();

    static void initCL(const char* kernel_source_path);

    static void setKernelParam(const char* kernel_key, int param_order, const char* buffer_key);

    static void createBuffer(const char* buffer_key, int buffer_size);

    static unsigned int writeBuffer(const char* buffer_key, unsigned int offset, std::vector<float> data);
    static unsigned int writeBuffer(const char* buffer_key, unsigned int offset, float data);

    static std::vector<float> readBuffer(const char* buffer_key, int offset, int size);

    static void createKernel(const char* kernel_key);
    static void queueKernel(const char* kernel_key, size_t global, size_t local);

private:
    static cl_int err;
    static cl_platform_id platform_id;
    static cl_device_id device_id;
    static cl_context context;
    static cl_command_queue command_queue;
    static cl_program program;

    static std::map<const char*, cl_kernel> kernels;
    static std::map<const char*, cl_mem> buffers;

    static std::string loadKernelSource(const char* filename);
};