#pragma once

#include <GL/glew.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <sstream>

class Graphics
{
public:
	Graphics();
	~Graphics();
	void draw();
	bool is_running();

private:
	const GLchar* loadShaderSource(const char* filename);
	GLint resolutionX, resolutionY;
	GLuint VBO, EBO;
	GLuint vertexShader, fragmentShader, shaderProgram;
	GLint modelLoc, viewLoc, projectionLoc;
	GLfloat cubePositionX;
	GLfloat projectionMatrix[4][4], viewMatrix[4][4];
	GLFWwindow *window;
};