#pragma once

#include "Neuron/Neuron.h"
#include "Layer/Layer.h"

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

	void loop();

	void train(GLuint cycles, GLuint epoch);
	void predict(GLfloat input1, GLfloat input2);
private:
	Layer inputLayer;
	Layer outputLayer;

	void draw();
	void initCL();
	std::string loadKernelSource(const char* filename);

	GLuint epoch, cyclesLeft;

	CLContainer clProgramInf;
};