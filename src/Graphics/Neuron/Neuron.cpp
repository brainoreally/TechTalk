#include "Neuron.h"

GLuint Neuron::modelUniformLocation = 0;
GLuint Neuron::colourUniformLocation = 0;
GLuint Neuron::VAO = 0;
GLuint Neuron::VBO = 0;
GLuint Neuron::EBO = 0;
GLuint Neuron::numIndices = 0;
glm::vec4 Neuron::colour = { 0.0f, 0.0f, 0.0f, 1.0f };

void Neuron::setupBuffers()
{
    // Define the vertex attributes
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    // Enable vertex attribute array and set the vertex attribute pointer
    glEnableVertexAttribArray(0);

    objectMeshData meshData = generateCubeMesh();

    // Define the vertex buffer object (VBO) and vertex array object (VAO)
    glGenBuffers(1, &VBO);
    // Bind the VBO and VAO to their respective buffer types
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    // Copy the vertices and indices into their respective buffers
    glBufferData(GL_ARRAY_BUFFER, meshData.vertices.size() * sizeof(std::tuple<GLfloat, GLfloat, GLfloat>), meshData.vertices.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    numIndices = meshData.indices.size();
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, numIndices * sizeof(std::tuple<GLuint, GLuint, GLuint>), meshData.indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
}

void Neuron::draw(glm::vec3 position, GLfloat value, std::vector<glm::vec3> oldPositions, std::vector<GLfloat> weights)
{
    // Set the model matrix
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, position);
    changeColour(value);
    glUniform4fv(colourUniformLocation, 1, &colour[0]);
    glUniformMatrix4fv(modelUniformLocation, 1, GL_FALSE, glm::value_ptr(model));

    // Bind the VAO and EBO and draw the sphere
    glBindVertexArray(VAO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glDrawElements(GL_TRIANGLE_STRIP, numIndices * 3, GL_UNSIGNED_INT, 0);


    for (int i = 0; i < oldPositions.size(); i++) {
        // Draw lines from old positions to the current position
        changeWeightColour(weights[i]);
        glUniform4fv(colourUniformLocation, 1, &colour[0]);
        glBegin(GL_LINES);
            // Set the line color for the connections
            glVertex3f(position.x, position.y * 0.75f, position.z);
            glVertex3f(oldPositions[i].x - position.x, oldPositions[i].y - position.y, oldPositions[i].z - position.z);
        glEnd();
    }
}

void Neuron::changeColour(GLfloat neuronValue)
{
    if (neuronValue >= 0.01f && neuronValue <= 0.99f) {
        colour = glm::vec4({ 1.0f * (1.0f - neuronValue), 1.0f * neuronValue, 0.0f, 1.0f });
    } else if(neuronValue < 0.01f) {
        colour = glm::vec4({ 1.0f, 0.0f, 0.0f, 1.0f });
    }
    else {
        colour = glm::vec4({ 0.0f, 1.0f, 0.0f, 1.0f });
    }
}

void Neuron::changeWeightColour(GLfloat weightValue)
{
    if (weightValue <= 0.1f) {
        colour = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
    }
    else if (weightValue <= 0.2f) {
        colour = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f);
    }
    else if (weightValue <= 0.3f) {
        colour = glm::vec4(1.0f, 0.5f, 0.0f, 1.0f);
    }
    else if (weightValue <= 0.4f) {
        colour = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
    }
    else if (weightValue <= 0.5f) {
        colour = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
    }
    else if (weightValue <= 0.6f) {
        colour = glm::vec4(0.0f, 1.0f, 1.0f, 1.0f);
    }
    else if (weightValue <= 0.7f) {
        colour = glm::vec4(0.5f, 0.0f, 1.0f, 1.0f);
    }
    else if (weightValue <= 0.8f) {
        colour = glm::vec4(0.5f, 1.0f, 0.0f, 1.0f);
    }
    else if (weightValue <= 0.9f) {
        colour = glm::vec4(0.0f, 0.5f, 1.0f, 1.0f);
    }
    else if (weightValue <= 1.0f) {
        colour = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);
    }
    else {
        colour = glm::vec4({ 0.6234f * weightValue, 0.0f, 1.0f * weightValue, 1.0f });
    }
}

objectMeshData Neuron::generateCubeMesh() {
    objectMeshData cubeMeshData;
    cubeMeshData.vertices = {
        // Front face
        {-0.5f, -0.5f, 0.5f }, // Vertex 0
        {-0.5f,  0.5f, 0.5f }, // Vertex 1
        { 0.5f,  0.5f, 0.5f }, // Vertex 2
        { 0.5f, -0.5f, 0.5f }, // Vertex 3
        // Back face
        {-0.5f, -0.5f, -0.5f }, // Vertex 4
        {-0.5f,  0.5f, -0.5f }, // Vertex 5
        { 0.5f,  0.5f, -0.5f }, // Vertex 6
        { 0.5f, -0.5f, -0.5f }, // Vertex 7
    };

    cubeMeshData.indices = {
        {0, 1, 2}, // Front face
        {2, 3, 0},
        {4, 5, 6}, // Back face
        {6, 7, 4},
        {1, 5, 6}, // Top face
        {6, 2, 1},
        {0, 4, 7}, // Bottom face
        {7, 3, 0},
        {3, 2, 6}, // Right face
        {6, 7, 3},
        {0, 1, 5}, // Left face
        {5, 4, 0},
    };
    
    return cubeMeshData;
}