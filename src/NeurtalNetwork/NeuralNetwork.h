#pragma once

#include "Neuron/Neuron.h"

#include <GL/glew.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <sstream>

class NeuralNetwork {
public:
	NeuralNetwork();
	~NeuralNetwork();

	void draw(GLfloat xOffset);

	void train(GLuint cycles);
	void predict(GLfloat input1, GLfloat input2);

private:
	Neuron inputLayer[2];
	Neuron outputLayer[1];
};