#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include <GL/glew.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>

#include <cmath>
#include <vector>
#include <tuple>
#include <map>
#include <iostream>

#include <CL/cl.h>
#include <iostream>

#include <vector>

#include "../Neuron/Neuron.h"

struct CLContainer {
	cl_int err;
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel forwardPassKernel;
	cl_kernel activateKernel;
	cl_kernel learnKernel;
	cl_mem outputBuffer;
	cl_mem correctOutputBuffer;
	cl_mem inputBuffer;
	cl_mem weightBuffer;
	cl_event event1, event2;
};

class Layer
{
public:
	Layer();
	Layer(int numNeurons, CLContainer clProgramInf);
	~Layer();

	void learn(GLfloat input1, GLfloat input2, GLfloat output, bool printEpoch);
	void draw(glm::vec3 position);
	void setNodeValues(std::vector<GLfloat> newValues);

	GLfloat forwardPass(GLfloat input1, GLfloat input2);

	std::vector<GLfloat> weights;
	std::vector<GLfloat> nodeValues;
private:
	GLfloat activate();
	GLfloat learningRate = 1.0f;
	GLfloat bias = 1.0f;

	static CLContainer clInf;

	int numNeurons;
};