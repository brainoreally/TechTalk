#pragma once

#include <GL/glew.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <sstream>

class NeuralNetwork {
public:
	NeuralNetwork();
	~NeuralNetwork();

	void train(GLuint cycles);
	void predict(GLfloat input1, GLfloat input2);

private:
	void learn(GLfloat input1, GLfloat input2, GLfloat output);
	
	GLfloat forwardPass(GLfloat input1, GLfloat input2);

	GLfloat learningRate = 1.0f;
	GLfloat bias = 1.0f;
	GLfloat weights[3];
};