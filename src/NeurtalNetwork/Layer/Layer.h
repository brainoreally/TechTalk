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

#include "../Neuron/Neuron.h"
#include "../CLProgram/CLProgram.h"

class Layer
{
public:
	Layer();
	Layer(int numNeurons);
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

	int numNeurons;
};