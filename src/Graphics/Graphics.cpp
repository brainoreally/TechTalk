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
    modelLoc = glGetUniformLocation(shaderProgram, "model");
    viewLoc = glGetUniformLocation(shaderProgram, "view");
    projectionLoc = glGetUniformLocation(shaderProgram, "projection");

    neuron = Neuron(modelLoc);
    initCL();
}

Graphics::~Graphics() {
    // Clean up
    clReleaseMemObject(buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    // Disable vertex attribute array
    glDisableVertexAttribArray(0);
    glUseProgram(0);
    glDeleteProgram(shaderProgram);
    glfwTerminate();
}

void Graphics::draw() {

    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


    // Set the view matrix
    glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 1.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

    // Set the projection matrix
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)resolutionX / (float)resolutionY, 0.1f, 100.0f);
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

    // Activate the shader program
    glUseProgram(shaderProgram);


    size_t global_size[1] = { 1 };
    size_t local_size[1] = { 1 };

    // Execute the kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_size, local_size, 0, NULL, NULL);

    // Copy the updated value of cubePositionX from the buffer
    err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, sizeof(float), &cubePositionX, 0, NULL, NULL);

    neuron.draw(glm::vec3(cubePositionX, 0.0f, -5.0f));
    neuron.draw(glm::vec3(cubePositionX, 1.5f, -5.0f));
    neuron.draw(glm::vec3(cubePositionX, -1.5f,-5.0f));

    // Swap buffers and poll events
    glfwSwapBuffers(window);
    glfwPollEvents();
}

bool Graphics::is_running() {
    return !glfwWindowShouldClose(window);
}

void Graphics::initCL()
{
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);

    // Compile and load the kernel
    const char* kernel_source =
        "__kernel void update_cube_position(__global float* cubePositionX)\n"
        "{\n"
        "    cubePositionX[0] += 0.01f;\n"
        "    if (cubePositionX[0] >= 2.0f)\n"
        "        cubePositionX[0] = -2.0f;\n"
        "}\n";
    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "update_cube_position", &err);

    // Create a buffer to hold the value of cubePositionX
    buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, &err);

    // Copy the value of cubePositionX to the buffer
    err = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, sizeof(float), &cubePositionX, 0, NULL, NULL);

    // Set the argument of the kernel to the buffer
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
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