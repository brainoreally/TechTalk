#include "Neuron.h"

Neuron::Neuron() {
    weights[0] = 0.2f;
    weights[1] = 0.3f;
    weights[2] = 0.5f;
}

Neuron::~Neuron()
{
}

GLuint Neuron::modelUniformLocation = 0;

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
    glUniformMatrix4fv(modelUniformLocation, 1, GL_FALSE, glm::value_ptr(model));

    // Draw the cube
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, 0);
}

GLfloat Neuron::forwardPass(GLfloat input1, GLfloat input2)
{
    GLfloat output = (input1 * weights[0]) + (input2 * weights[1]) + (bias * weights[2]);
    return heavySideActivation(output);
}

void Neuron::learn(GLfloat input1, GLfloat input2, GLfloat output) {
    GLfloat outputP = forwardPass(input1, input2);

    GLfloat error = output - outputP;
    weights[0] += error * input1 * learningRate;
    weights[1] += error * input2 * learningRate;
    weights[2] += error * bias * learningRate;
}

GLfloat Neuron::heavySideActivation(GLfloat neuronOutput)
{
    if (neuronOutput > 0.0f)
        neuronOutput = 1.0f;
    else
        neuronOutput = 0.0f;

    return neuronOutput;
}