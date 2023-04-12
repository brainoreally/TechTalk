#include "Graphics.h"

Graphics::Graphics() {
    // Define screen size
    resolutionX = 1800;
    resolutionY = 1200;

    // Initialize cube X position
    cubePositionX = -2.0f;

    // Define the vertices and indices of the cube
    GLfloat vertices[] = {
        // Front face
        -0.5f, -0.5f, 0.5f, // Vertex 0
        -0.5f, 0.5f, 0.5f,  // Vertex 1
        0.5f, 0.5f, 0.5f,   // Vertex 2
        0.5f, -0.5f, 0.5f,  // Vertex 3
        // Back face
        -0.5f, -0.5f, -0.5f, // Vertex 4
        -0.5f, 0.5f, -0.5f,  // Vertex 5
        0.5f, 0.5f, -0.5f,   // Vertex 6
        0.5f, -0.5f, -0.5f,  // Vertex 7
    };

    GLushort indices[] = {
        0, 1, 2,  // Front face
        2, 3, 0,
        4, 5, 6,  // Back face
        6, 7, 4,
        1, 5, 6, // Top face
        6, 2, 1,
        0, 4, 7, // Bottom face
        7, 3, 0,
        3, 2, 6, // Right face
        6, 7, 3,
        0, 1, 5, // Left face
        5, 4, 0,
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

    // Compile and link the shader program
    std::string vertexSourceCode = loadShaderSource("src/shaders/vertex.glsl");
    const GLchar* vertexShaderSource = vertexSourceCode.c_str();
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    std::string fragmentShaderCode = loadShaderSource("src/shaders/fragment.glsl");
    const GLchar* fragmentShaderSource = fragmentShaderCode.c_str();
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Check if the linking was successful
    GLint linkStatus;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &linkStatus);
    if (linkStatus != GL_TRUE) {
        // Linking failed, get the error message
        GLint infoLogLength;
        glGetProgramiv(shaderProgram, GL_INFO_LOG_LENGTH, &infoLogLength);
        char* infoLog = new char[infoLogLength];
        glGetProgramInfoLog(shaderProgram, infoLogLength, NULL, infoLog);
        std::cerr << "Shader program linking failed: " << infoLog << std::endl;
        delete[] infoLog;
        exit(EXIT_FAILURE);
    }

    // delete shaders
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Set up the uniform locations
    modelLoc = glGetUniformLocation(shaderProgram, "model");
    viewLoc = glGetUniformLocation(shaderProgram, "view");
    projectionLoc = glGetUniformLocation(shaderProgram, "projection");

    // Define the vertex attributes
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    // Enable vertex attribute array and set the vertex attribute pointer
    glEnableVertexAttribArray(0);

    // Define the vertex buffer object (VBO) and vertex array object (VAO)
    glGenBuffers(1, &VBO);
    // Bind the VBO and VAO to their respective buffer types
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    // Copy the vertices and indices into their respective buffers
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
}

Graphics::~Graphics() {
    // Disable vertex attribute array
    glDisableVertexAttribArray(0);
    glUseProgram(0);
    glDeleteProgram(shaderProgram);
    glfwTerminate();
}

void Graphics::draw() {
    cubePositionX += 0.01f;
    if (cubePositionX >= 2.0f)
        cubePositionX = -2.0f;

    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Set the model matrix
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(cubePositionX, 0.0f, 0.0f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

    // Set the view matrix
    glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 1.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

    // Set the projection matrix
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)resolutionX / (float)resolutionY, 0.1f, 100.0f);
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

    // Activate the shader program
    glUseProgram(shaderProgram);

    // Draw the cube
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, 0);

    // Swap buffers and poll events
    glfwSwapBuffers(window);
    glfwPollEvents();
}

bool Graphics::is_running() {
    return !glfwWindowShouldClose(window);
}

std::string Graphics::loadShaderSource(const char* filename)
{
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return "";
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    
    file.close();
    
    return buffer.str();
}