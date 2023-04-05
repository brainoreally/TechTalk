#include "Graphics.h"

Graphics::Graphics() {
    // Define screen size
    resolutionX = 800;
    resolutionY = 600;

    // Initialize cube X position
    cubePositionX = 0.0f;

    // Define the vertices of the cube
    GLfloat vertices[] = {
        -0.5f, -0.5f, -0.5f,
        -0.5f, 0.5f, -0.5f,
        0.5f, 0.5f, -0.5f,
        0.5f, -0.5f, -0.5f,
        -0.5f, -0.5f, 0.5f,
        -0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f,
        0.5f, -0.5f, 0.5f
    };

    // Define the faces of the cube
    GLubyte indices[] = {
        0, 1, 2, 3,
        3, 2, 6, 7,
        7, 6, 5, 4,
        4, 5, 1, 0,
        5, 6, 2, 1,
        7, 4, 0, 3
    };

    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return;
    }

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(resolutionX, resolutionY, "Blue Cube", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        glfwTerminate();
        return;
    }

    // Set up the viewport
    glViewport(0, 0, resolutionX, resolutionY);

    // Set up the projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60, 1, 0.1, 100);

    // Set up the view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0);

    // Define the vertex buffer object (VBO) and element buffer object (EBO)
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    // Bind the VBO and EBO to their respective buffer types
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

    // Copy the vertices and indices into their respective buffers
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Define the vertex attributes
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);
}

Graphics::~Graphics() {
    glfwTerminate();
    delete window;
}

void Graphics::draw() {
    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT);

    // Bind the VBO and EBO to their respective buffer types
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

    // Enable vertex attribute array and set the vertex attribute pointer
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);

    // Update the cubes position
    cubePositionX += 0.01f;
    if (cubePositionX > 2.0f) cubePositionX = 0.0f;

    // Traslate to draw cube in it's world position
    glTranslatef(cubePositionX, 0, 0);

    // Draw the cube
    glDrawElements(GL_QUADS, 24, GL_UNSIGNED_BYTE, (GLvoid*)0);

    // Translate back to the camera position
    glTranslatef(-cubePositionX, 0, 0);

    // Swap front and back buffers
    glfwSwapBuffers(window);
}

bool Graphics::isRunning() {
    return !glfwWindowShouldClose(window);
}