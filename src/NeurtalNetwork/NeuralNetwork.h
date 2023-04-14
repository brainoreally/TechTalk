#pragma once

#include "Neuron/Neuron.h"

#include <GL/glew.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <sstream>
#include <fstream>

class NeuralNetwork {
public:
	NeuralNetwork();
	~NeuralNetwork();

	void draw();

	void train(GLuint cycles, GLuint epoch);
	void predict(GLfloat input1, GLfloat input2);

private:
	Neuron inputLayer[2];
	Neuron outputLayer[1];

	void initCL();
	std::string loadKernelSource(const char* filename);

	cl_int err;
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem outputBuffer;
	cl_mem inputBuffer;
	cl_mem weightBuffer;
};