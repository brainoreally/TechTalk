#include <chrono>  // for timing
#include <thread>  // for sleeping
#include "src/Graphics/Graphics.h"
#include "src/NeurtalNetwork/NeuralNetwork.h"

int main() {

    Graphics graphics = Graphics();
    NeuralNetwork perceptronNetwork = NeuralNetwork();

    GLuint iterations = 5000;
    perceptronNetwork.train(iterations, iterations / 10);

    int iter = 0;
    GLfloat combinations[4][2] = {
        { 0.0f, 0.0f },
        { 1.0f, 0.0f },
        { 0.0f, 1.0f },
        { 1.0f, 1.0f }
    };

    auto last_prediction_time = std::chrono::high_resolution_clock::now();  // initialize timer

    // Main loop
    while (graphics.is_running()) {

        // Check if 0.5 seconds have passed since last prediction
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_prediction_time).count() >= 500) {
            perceptronNetwork.predict(combinations[iter][0], combinations[iter][1]);
            iter = (iter + 1) % 4;
            last_prediction_time = now;  // update last prediction time
        }

        graphics.setupScene();
        perceptronNetwork.loop();
        graphics.swapBuffersAndPoll();

        std::this_thread::sleep_for(std::chrono::milliseconds(16));  // limit frame rate to ~60 fps
    }

    // Clean up
    return 0;
}
