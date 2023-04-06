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
	bool is_running();

private:
	GLint resolutionX, resolutionY;
	GLuint VBO, EBO, VAO;
	GLfloat cubePositionX;
	GLFWwindow* window;
};