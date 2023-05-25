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
	void drawNeurons(std::vector<std::vector<GLfloat>> neuronValues, std::vector<std::vector<GLfloat>> weights, std::vector<std::vector<GLfloat>> biases);
private:
	std::string loadShaderSource(const char* filename);
	GLint resolutionX, resolutionY;
	GLuint neuronVertexShader, neuronFragmentShader, neuronShaderProgram;
	GLint modelLoc, viewLoc, projectionLoc;
	GLfloat cubePositionX;
	GLFWwindow *window;
};