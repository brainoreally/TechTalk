#include "src/Graphics/Graphics.h"

GLfloat learningRate = 1.00f;
GLfloat bias = 1.0f;
GLfloat weights[3] = { 0.3f, 0.5f, 0.7f };

void learn(GLfloat input1, GLfloat input2, GLfloat output) {
    GLfloat outputP = (input1 * weights[0]) + (input2 * weights[1]) + (bias * weights[2]);
    if (outputP > 0.0f)
        outputP = 1.0f;
    else
        outputP = 0.0f;

    GLfloat error = output - outputP;
    weights[0] += error * input1 * learningRate;
    weights[0] += error * input2 * learningRate;
    weights[0] += error * bias * learningRate;
}

void predict(GLfloat input1, GLfloat input2) {
    GLfloat outputP = (input1 * weights[0]) + (input2 * weights[1]) + (bias * weights[2]);
    if (outputP > 0.0f)
        outputP = 1.0f;
    else
        outputP = 0.0f;

    std::cout << "Output for values (" + std::to_string(input1) + ", " + std::to_string(input2) + ") is: " + std::to_string(outputP) << std::endl;
}

int main() {

    //Graphics graphics = Graphics();

    for (int i = 0; i < 50; i++) {
        learn(1.0f, 1.0f, 1.0f); // True  or True  = True
        learn(1.0f, 0.0f, 1.0f); // True  or False = True
        learn(0.0f, 1.0f, 1.0f); // False or True  = True
        learn(0.0f, 0.0f, 1.0f); // False or False = False
    }

    predict(0.0f, 0.0f);
    predict(1.0f, 0.0f);
    predict(0.0f, 1.0f);
    predict(1.0f, 1.0f);

    // Main loop
    //while (graphics.is_running()) {
        //graphics.draw();
    //}

    // Clean up
    return 0;
}