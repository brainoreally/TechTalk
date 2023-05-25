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

struct objectMeshData {
	std::vector<std::tuple<GLfloat, GLfloat, GLfloat>> vertices;
	std::vector<std::tuple<GLuint, GLuint, GLuint>> indices;
};

class Neuron
{
public:
	static void draw(glm::vec3 position, GLfloat value, std::vector<glm::vec3> oldPositions, std::vector<GLfloat> weights);
	static void changeColour(GLfloat neuronValue);
	static void changeWeightColour(GLfloat weightValue);

	static GLuint colourUniformLocation;
	static GLuint modelUniformLocation;
	static void setupBuffers();
private:
	static glm::vec4 colour;

	static GLuint VBO, EBO, VAO, numIndices;
	static objectMeshData generateCubeMesh();
};