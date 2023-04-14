#pragma once

#include "..\NeurtalNetwork\NeuralNetwork.h"
#include <GL/glew.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <sstream>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include <CL/cl.h>

class Graphics
{
public:
	Graphics();
	~Graphics();
	void draw(NeuralNetwork * perceptronNetwork);
	bool is_running();

private:
	void initCL();

	std::string loadShaderSource(const char* filename);
	GLint resolutionX, resolutionY;
	GLuint vertexShader, fragmentShader, shaderProgram;
	GLint modelLoc, viewLoc, projectionLoc;
	GLfloat cubePositionX;
	GLFWwindow *window;

	cl_int err;
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem buffer;
};