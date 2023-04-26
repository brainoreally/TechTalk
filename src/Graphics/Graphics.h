#pragma once

#include "Neuron/Neuron.h"

#include <fstream>
#include <sstream>

class Graphics
{
public:
	Graphics();
	~Graphics();
	void setupScene();
	void swapBuffersAndPoll();
	bool is_running();
	void drawNeurons(std::vector<std::vector<GLfloat>> neuronValues);
private:
	std::string loadShaderSource(const char* filename);
	GLint resolutionX, resolutionY;
	GLuint vertexShader, fragmentShader, shaderProgram;
	GLint modelLoc, viewLoc, projectionLoc;
	GLfloat cubePositionX;
	GLFWwindow *window;
};