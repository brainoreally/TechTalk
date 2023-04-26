#include <chrono>  // for timing
#include <thread>  // for sleeping
#include "src/Graphics/Graphics.h"
#include "src/NeurtalNetwork/NeuralNetwork.h"

NetworkParams buildPerceptronParams() {
    NetworkParams out = NetworkParams(2, 1, 1, 1);

    std::map<const char*, const char*> inputLayerKernelKeys = {
        { "forward_pass", "forward_pass" },
        { "activate", "activate" },
        { "learn", "learn" },
    };
    std::map<const char*, const char*> inputLayerBufferKeys = {
        { "inputs", "inputs" },
        { "weights", "weights" },
        { "correctOutput", "correctOutput" },
    };
    out.inputLayerParams = LayerParams(2, inputLayerKernelKeys, inputLayerBufferKeys);
    out.outputLayerParams = LayerParams(2, {}, { {"output", "output"} });

    return out;
}

int main() {

    Graphics graphics = Graphics();

    NetworkParams perceptronNetworkParams = buildPerceptronParams();
    NeuralNetwork perceptronNetwork = NeuralNetwork(perceptronNetworkParams);

    std::vector<std::vector<std::vector<float>>> trainingData = {
        { { 0.0f, 0.0f }, { 0.0f }, },
        { { 1.0f, 0.0f }, { 1.0f }, },
        { { 0.0f, 1.0f }, { 1.0f }, },
        { { 1.0f, 1.0f }, { 1.0f }, },
    };

    GLuint iterations = 5000;
    perceptronNetwork.train(trainingData, iterations, iterations / 10);

    int iter = 0;
    auto last_prediction_time = std::chrono::high_resolution_clock::now();  // initialize timer

    // Main loop
    while (graphics.is_running()) {

        // Check if 0.5 seconds have passed since last prediction
        auto now = std::chrono::high_resolution_clock::now();
        if (!perceptronNetwork.training && std::chrono::duration_cast<std::chrono::milliseconds>(now - last_prediction_time).count() >= 500) {
            perceptronNetwork.predict(trainingData[iter][0]);
            iter = (iter + 1) % 4;
            last_prediction_time = now;  // update last prediction time
        }

        graphics.setupScene();

        graphics.drawNeurons(perceptronNetwork.returnNetworkValues());

        graphics.swapBuffersAndPoll();

        std::this_thread::sleep_for(std::chrono::milliseconds(16));  // limit frame rate to ~60 fps
    }

    // Clean up
    return 0;
}
