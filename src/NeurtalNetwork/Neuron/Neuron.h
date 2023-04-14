#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include <GL/glew.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>

class Neuron
{
public:
	Neuron();
	~Neuron();

	void learn(GLfloat input1, GLfloat input2, GLfloat output);
	void draw(glm::vec3 position);

	GLfloat forwardPass(GLfloat input1, GLfloat input2);

	static GLuint modelUniformLocation;
	static void setupBuffers();
private:

	GLfloat heavySideActivation(GLfloat layerOutput);

	GLfloat learningRate = 1.0f;
	GLfloat bias = 1.0f;
	GLfloat weights[3];
};