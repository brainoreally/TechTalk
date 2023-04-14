#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include <GL/glew.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>

#include <cmath>

#include <CL/cl.h>

class Neuron
{
public:
	Neuron();
	Neuron(cl_command_queue* q, cl_kernel* kern, cl_mem* outpBuffer, cl_mem* inpBuffer, cl_mem* weigBuffer);
	~Neuron();

	void learn(GLfloat input1, GLfloat input2, GLfloat output);
	void draw(glm::vec3 position);

	GLfloat forwardPass(GLfloat input1, GLfloat input2);

	static GLuint modelUniformLocation;
	static void setupBuffers();
	GLfloat weights[3];
private:
	GLfloat sigmoidActivation(GLfloat layerOutput);

	GLfloat learningRate = 1.0f;
	GLfloat bias = 1.0f;

	cl_command_queue* queue;
	cl_kernel* kernel;
	cl_mem* outputBuffer;
	cl_mem* inputBuffer;
	cl_mem* weightBuffer;
};