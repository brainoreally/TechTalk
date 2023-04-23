#include "Graphics.h"

Graphics::Graphics() {
    // Define screen size
    resolutionX = 1800;
    resolutionY = 1200;

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
    Neuron::modelUniformLocation = glGetUniformLocation(shaderProgram, "model");
    Neuron::colourUniformLocation = glGetUniformLocation(shaderProgram, "colour");
    viewLoc = glGetUniformLocation(shaderProgram, "view");
    projectionLoc = glGetUniformLocation(shaderProgram, "projection");

    Neuron::setupBuffers();
 }

Graphics::~Graphics() {
    // Disable vertex attribute array
    glDisableVertexAttribArray(0);
    glUseProgram(0);
    glDeleteProgram(shaderProgram);
    glfwTerminate();
}

void Graphics::setupScene() {

    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Set the view matrix
    glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

    // Set the projection matrix
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)resolutionX / (float)resolutionY, 0.1f, 100.0f);
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

    // Activate the shader program
    glUseProgram(shaderProgram);
}

void Graphics::swapBuffersAndPoll(){
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