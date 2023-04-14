#include "src/Graphics/Graphics.h"
#include "src/NeurtalNetwork/NeuralNetwork.h"

int main() {

    Graphics graphics = Graphics();
    NeuralNetwork perceptronNetwork = NeuralNetwork();

    perceptronNetwork.train(5000);

    perceptronNetwork.predict(0.0f, 0.0f);
    perceptronNetwork.predict(1.0f, 0.0f);
    perceptronNetwork.predict(0.0f, 1.0f);
    perceptronNetwork.predict(1.0f, 1.0f);

    // Main loop
    while (graphics.is_running()) {
        graphics.setupScene();
        perceptronNetwork.draw();
        graphics.swapBuffersAndPoll();
    }

    // Clean up
    return 0;
}