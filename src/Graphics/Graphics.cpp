#include "Graphics.h"
#include <Windows.h>

Graphics::Graphics() {
    // Get the handle to the console window
    HWND consoleWindow = GetConsoleWindow();

    // Set the position of the console window
    SetWindowPos(consoleWindow, NULL, 0, 50, 0, 0, SWP_NOSIZE | SWP_NOZORDER);

    // Define screen size
    resolutionX = 2400;
    resolutionY = 2000;

    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return;
    }

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(resolutionX, resolutionY, "Blue Cube", NULL, NULL);
    glfwSetWindowPos(window, 1200, 50);
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
    const GLchar* neuronVertexShaderSource = vertexSourceCode.c_str();
    neuronVertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(neuronVertexShader, 1, &neuronVertexShaderSource, NULL);
    glCompileShader(neuronVertexShader);

    std::string neuronFragmentShaderCode = loadShaderSource("src/shaders/fragment.glsl");
    const GLchar* neuronFragmentShaderSource = neuronFragmentShaderCode.c_str();
    neuronFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(neuronFragmentShader, 1, &neuronFragmentShaderSource, NULL);
    glCompileShader(neuronFragmentShader);

    neuronShaderProgram = glCreateProgram();
    glAttachShader(neuronShaderProgram, neuronVertexShader);
    glAttachShader(neuronShaderProgram, neuronFragmentShader);
    glLinkProgram(neuronShaderProgram);

    // Check if the linking was successful
    GLint linkStatus;
    glGetProgramiv(neuronShaderProgram, GL_LINK_STATUS, &linkStatus);
    if (linkStatus != GL_TRUE) {
        // Linking failed, get the error message
        GLint infoLogLength;
        glGetProgramiv(neuronShaderProgram, GL_INFO_LOG_LENGTH, &infoLogLength);
        char* infoLog = new char[infoLogLength];
        glGetProgramInfoLog(neuronShaderProgram, infoLogLength, NULL, infoLog);
        std::cerr << "Shader program linking failed: " << infoLog << std::endl;
        delete[] infoLog;
        exit(EXIT_FAILURE);
    }

    // delete shaders
    glDeleteShader(neuronVertexShader);
    glDeleteShader(neuronFragmentShader);

    // Set up the uniform locations
    Neuron::modelUniformLocation = glGetUniformLocation(neuronShaderProgram, "model");
    Neuron::colourUniformLocation = glGetUniformLocation(neuronShaderProgram, "colour");
    viewLoc = glGetUniformLocation(neuronShaderProgram, "view");
    projectionLoc = glGetUniformLocation(neuronShaderProgram, "projection");

    glEnable(GL_DEPTH_TEST);  // Enable depth testing

    glDepthFunc(GL_LESS);  // Set depth function to GL_LESS

    Neuron::setupBuffers();
 }

Graphics::~Graphics() {
    // Disable vertex attribute array
    glDisableVertexAttribArray(0);
    glUseProgram(0);
    glDeleteProgram(neuronShaderProgram);
    glfwTerminate();
}

void Graphics::setupScene() {

    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Set the view matrix
    glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

    // Set the projection matrix
    glm::mat4 projection = glm::perspective(glm::radians(90.0f), (float)resolutionX / (float)resolutionY, 0.1f, 20000.0f);
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

    // Activate the shader program
    glUseProgram(neuronShaderProgram);
}

void Graphics::swapBuffersAndPoll(){
    // Swap buffers and poll events
    glfwSwapBuffers(window);
    glfwPollEvents();
}

bool Graphics::is_running() {
    return !glfwWindowShouldClose(window);
}

void Graphics::drawNeurons(std::vector<std::vector<GLfloat>> networkValues, std::vector<std::vector<GLfloat>> weights, std::vector<std::vector<GLfloat>> biases)
{
    GLfloat xPos = networkValues.size() * -0.75f;
    GLfloat zPos = networkValues.size() * -2.0f;
    
    if (networkValues[0].size() == 28 * 28) {
        int iter = 0;
        for (int y = 0; y < 28; y++) {
            GLfloat xOff = -70.0f;
            GLfloat yPos = 20.0f - (1.5f * y);
            GLfloat zOff = -50.0f;
            for (int x = 0; x < 28; x++) {

                Neuron::draw(glm::vec3(xOff, yPos, zPos + zOff), networkValues[0][iter], 0, {}, {});
                xOff += 1.5f;
                ++iter;
            }
        }
    }

    std::vector<glm::vec3> oldPositions = {};
    std::vector<glm::vec3> newPositions = {};
    for (int i = 0; i < networkValues.size() - 1; i++) {
        GLfloat yPos = networkValues[i].size() * -1.0f;
        oldPositions = newPositions;
        newPositions = {};

        std::vector<GLfloat> weightVals = {};
        std::vector<GLfloat> biasVals = {};
        if (i > 0) {
            weightVals = weights[i - 1];
            biasVals = biases[i - 1];
        }
        for (int neuronIter = 0; neuronIter < networkValues[i].size(); neuronIter++) {
            glm::vec3 pos = glm::vec3(xPos, yPos, zPos);
            newPositions.push_back(pos);
            int bias = 0;
            if (i > 0)
                bias = biasVals[neuronIter];
            Neuron::draw(pos, networkValues[i][neuronIter], bias, oldPositions, weightVals);

            yPos += 2.0f;
        }
        xPos += 2.0f;
    }

    xPos = 6.0f;
    zPos = -2.0f;
    GLfloat yPos = networkValues[networkValues.size() - 1].size() * -0.75f;
    for (GLfloat neuronValue : networkValues[networkValues.size() - 1]) {
        Neuron::draw(glm::vec3(xPos, yPos, zPos), neuronValue, 0, {}, {});
        yPos += 2.0f;
    }
    xPos += 2.0f;
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