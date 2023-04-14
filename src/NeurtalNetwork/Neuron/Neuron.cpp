#include "Neuron.h"

Neuron::Neuron()
{
}

Neuron::Neuron(cl_command_queue* q, cl_kernel* kern, cl_mem* outpBuffer, cl_mem* inpBuffer, cl_mem* weigBuffer) {
    weights[0] = 0.2f;
    weights[1] = 0.3f;
    weights[2] = 0.5f;

    queue = q;
    kernel = kern;
    outputBuffer = outpBuffer;
    inputBuffer = inpBuffer;
    weightBuffer = weigBuffer;
}

Neuron::~Neuron()
{
}

GLuint Neuron::modelUniformLocation = 0;
GLuint Neuron::colourUniformLocation = 0;

void Neuron::setupBuffers()
{
    GLuint VBO, EBO, VAO;
    // Define the vertex attributes
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    // Enable vertex attribute array and set the vertex attribute pointer
    glEnableVertexAttribArray(0);

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

void Neuron::draw(glm::vec3 position)
{
    // Set the model matrix
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, position);
    glUniform4fv(colourUniformLocation, 1, &colour[0]);
    glUniformMatrix4fv(modelUniformLocation, 1, GL_FALSE, glm::value_ptr(model));

    // Draw the cube
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, 0);
}

void Neuron::changeColour(GLfloat output)
{
    if (output < 0.1)
        colour = glm::vec4({ 1.0f, 0.0f, 0.0f, 1.0f });
    else
        colour = glm::vec4({ 0.0f, 1.0f, 0.0f, 1.0f });
}

GLfloat Neuron::forwardPass(GLfloat input1, GLfloat input2)
{
    // Copy the value of input1 and input2 to the buffer
    cl_int err = clEnqueueWriteBuffer(*queue, *inputBuffer, CL_TRUE, 0, sizeof(GLfloat), &input1, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(*queue, *inputBuffer, CL_TRUE, sizeof(GLfloat), sizeof(GLfloat), &input2, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(*queue, *weightBuffer, CL_TRUE, 0, 3 * sizeof(GLfloat), &weights, 0, NULL, NULL);

    size_t global_size[1] = { 1 };
    size_t local_size[1] = { 1 };

    // Execute the kernel
    err = clEnqueueNDRangeKernel(*queue, *kernel, 1, NULL, global_size, local_size, 0, NULL, NULL);
    // Copy the output from the buffer
    GLfloat output = 0.0f;
    err = clEnqueueReadBuffer(*queue, *outputBuffer, CL_TRUE, 0, sizeof(float), &output, 0, NULL, NULL);

    return output;
}

void Neuron::learn(GLfloat input1, GLfloat input2, GLfloat output, bool printEpoch) {
    GLfloat outputP = forwardPass(input1, input2);

    GLfloat error = output - outputP;
    weights[0] += error * input1 * learningRate;
    weights[1] += error * input2 * learningRate;
    weights[2] += error * bias * learningRate;

    if (printEpoch) {
        std::cout << "    Testing (" << input1 << ", " << input2 << "): { Output: " << outputP <<", Expected: " << output << ", Error: " << error << " }" << std::endl;
    }
}

GLfloat Neuron::sigmoidActivation(GLfloat neuronOutput)
{
    return neuronOutput = 1.0f / (1.0f + std::exp(-neuronOutput));
}