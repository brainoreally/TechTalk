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

class Graphics
{
public:
	Graphics();
	~Graphics();
	void setupScene();
	void swapBuffersAndPoll();
	bool is_running();

private:
	std::string loadShaderSource(const char* filename);
	GLint resolutionX, resolutionY;
	GLuint vertexShader, fragmentShader, shaderProgram;
	GLint modelLoc, viewLoc, projectionLoc;
	GLfloat cubePositionX;
	GLFWwindow *window;
};