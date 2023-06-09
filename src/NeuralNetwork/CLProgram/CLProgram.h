#pragma once

#include <CL/cl.h>
#include <string>

#include "../ParamStructs.h"

class CLProgram {
public:
    static void cleanup();
    static void initCL(const char* kernel_source_path);
    
    static void createKernel(const char* kernel_key);
    static void setKernelParam(const char* kernel_key, int param_order, const char* buffer_key);
    static void queueKernel(const char* kernel_key, size_t global, size_t local);
    static void setupNetworkOpenCL(NetworkParams* params);

    template<typename Datatype>
    static void createBuffer(const char* buffer_key, int buffer_size) {
        buffers[buffer_key] = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size * sizeof(Datatype), NULL, &err);
    }

    template<typename DataType>
    static unsigned int writeBuffer(const char* buffer_key, unsigned int offset, std::vector<std::vector<DataType>>& data) {
        size_t numRows = data.size();
        size_t numCols = (numRows > 0) ? data[0].size() : 0;
        size_t bufferSize = numRows * numCols * sizeof(DataType);
        std::vector<DataType> test;
        for (int x = 0; x < numRows; x++) {
            for (int y = 0; y < numCols; y++) {
                test.push_back(data[x][y]);
            }
        }

        return writeBuffer<DataType>(buffer_key, offset, test);
    }


    template<typename Datatype>
    static unsigned int writeBuffer(const char* buffer_key, unsigned int offset, std::vector<Datatype> data) {
        clEnqueueWriteBuffer(command_queue, buffers[buffer_key], CL_TRUE, offset, data.size() * sizeof(Datatype), &data[0], 0, NULL, NULL);
        unsigned int newOffset = offset + (data.size() * sizeof(Datatype));
        return newOffset;
    }

    template<typename Datatype>
    static unsigned int writeBuffer(const char* buffer_key, unsigned int offset, Datatype data) {
        clEnqueueWriteBuffer(command_queue, buffers[buffer_key], CL_TRUE, offset, sizeof(Datatype), &data, 0, NULL, NULL);
        unsigned int newOffset = offset + sizeof(Datatype);
        return newOffset;
    }

    template<typename Datatype>
    static std::vector<Datatype> readBuffer(const char* buffer_key, int offset, int size)
    {
        std::vector<Datatype> output;
        for(int i = 0; i < size; i++)
            output.push_back(0.0f);
        err = clEnqueueReadBuffer(command_queue, buffers[buffer_key], CL_TRUE, offset * sizeof(Datatype), size * sizeof(Datatype), &output[0], 0, NULL, NULL);
        return output;
    }

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