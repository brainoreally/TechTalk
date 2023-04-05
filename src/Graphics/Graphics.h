#pragma once

#include <GL/glew.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include <iostream>

class Graphics
{
public:
	Graphics();
	~Graphics();
	void draw();
	bool isRunning();

private:
	GLint resolutionX, resolutionY;
	GLuint VBO, EBO;
	GLfloat cubePositionX;
	GLFWwindow* window;
};